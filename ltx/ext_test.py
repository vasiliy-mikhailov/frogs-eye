"""Experiment: continue a clip with the `ext` graph through raw ComfyUI, stitch it, and measure the seam.

Run on the Halo in the ltx image, sharing the ltx container's network and data volume:
  docker run --rm --network container:ltx -v ltx-data:/data -v C:\\ltx\\build:/b:ro --entrypoint python \\
    ltx-comfy:v0.45.0 /b/ext_test.py <source mp4 under /data> <new seconds> "<prompt>" [audio_from: 1|2]
Writes /data/out/ltx/exttest_*.mp4 and seam frame PNGs to /data/out/ltx/seam/, and prints seam metrics.
"""
import json, subprocess, sys, time, urllib.request, uuid, wave

import numpy as np

B = "http://127.0.0.1:8188"
import os
src, new_s = sys.argv[1], float(sys.argv[2])
prompt = os.environ.get("EXT_PROMPT") or sys.argv[3]
audio_from = sys.argv[4] if len(sys.argv) > 4 else "2"
OVERLAP = 121


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return int(st["width"]), int(st["height"]), int(st["nb_read_frames"]), float(num) / float(den)


def call(path, data=None):
    req = urllib.request.Request(B + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


W, H, N, fps = probe(src)
start = N - OVERLAP
print("source %dx%d, %d frames at %g fps; overlap starts at frame %d" % (W, H, N, fps, start), flush=True)
t0 = time.time()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "%.6f" % (start / fps), "-i", src, "-frames:v", str(OVERLAP),
                "-t", "%.6f" % (OVERLAP / fps), "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-b:a", "256k", "/data/input/exttest_overlap.mp4"], check=True)
print("tail cut in %.1f s; tail frames: %d" % (time.time() - t0, probe("/data/input/exttest_overlap.mp4")[2]), flush=True)

new_frames = int(round(new_s * fps / 8)) * 8
g = json.load(open("/b/adapter/ltx25_distilled_gguf_ext_api.json"))
g["372"]["inputs"]["value"], g["360"]["inputs"]["value"] = W, H
g["362"]["inputs"]["value"], g["361"]["inputs"]["value"] = OVERLAP + new_frames, int(round(fps))
g["339"]["inputs"]["noise_seed"] = 4242
g["364"]["inputs"]["text"] = prompt
g["500"]["inputs"]["file"] = "exttest_overlap.mp4"
g["75"]["inputs"]["filename_prefix"] = "ltx/exttest_raw"
if audio_from == "1":
    g["358"]["inputs"]["samples"] = ["367", 1]  # decode the stage-1 audio (pinned overlap kept exactly)
t0 = time.time()
pid = call("/prompt", {"prompt": g, "client_id": str(uuid.uuid4())})["prompt_id"]
print("submitted %d frames (%d overlap + %d new), audio from stage %s" % (OVERLAP + new_frames, OVERLAP, new_frames, audio_from), flush=True)
while True:
    time.sleep(3)
    h = call("/history/" + pid)
    if pid in h:
        break
st = h[pid]["status"]
if st.get("status_str") == "error":
    print("ERROR", json.dumps([m for m in st.get("messages", []) if m[0] == "execution_error"])[:3000]); sys.exit(1)
out = next(f for o in h[pid]["outputs"].values() for v in o.values() if isinstance(v, list) for f in v
           if isinstance(f, dict) and f.get("type") == "output" and str(f.get("filename", "")).endswith(".mp4"))
raw = "/data/out/%s/%s" % (out["subfolder"], out["filename"])
print("rendered in %.0f s -> %s (%d frames)" % (time.time() - t0, raw, probe(raw)[2]), flush=True)

# stitch: source + new frames after the overlap; audio likewise
t0 = time.time()
cut = OVERLAP / fps
stitched = "/data/out/ltx/exttest_stitched.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-i", raw, "-filter_complex",
                "[0:v]settb=1/{f},setpts=N[v0];[1:v]trim=start_frame={ov},settb=1/{f},setpts=N[v1];[v0][v1]concat=n=2:v=1:a=0[v];"
                "[0:a]apad,atrim=end={sd:.6f}[a0];[1:a]apad,atrim=start={s1:.6f},asetpts=PTS-STARTPTS[a1];[a0][a1]acrossfade=d=0.1[a]"
                .format(f=round(fps), ov=OVERLAP, sd=N / fps, s1=cut - 0.1),
                "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-fps_mode", "cfr", "-r", str(round(fps)), "-c:a", "aac", "-b:a", "192k", "-t", "%.6f" % ((N + new_frames) / fps),
                stitched], check=True)  # same timing as the adapter's stitch()
print("stitched in %.1f s -> %s: %d frames (expected %d)" % (time.time() - t0, stitched, probe(stitched)[2], N + new_frames), flush=True)


def frames(path, first, count):
    data = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "select=between(n\\,%d\\,%d)" % (first, first + count - 1),
                           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(data, np.uint8).reshape(-1, H, W, 3).astype(np.float32)


# 1) overlap fidelity: the regenerated overlap vs the source tail it was pinned to
src_tail, new_head = frames(src, start, OVERLAP), frames(raw, 0, OVERLAP + 3)
fid = [float(np.abs(src_tail[i] - new_head[i]).mean()) for i in (0, 60, OVERLAP - 1)]
print("overlap fidelity, mean |src - new| (0-255) at overlap frames 0/60/120: %.2f / %.2f / %.2f" % tuple(fid))
# 2) seam continuity: the jump at the cut vs ordinary frame-to-frame motion around it
around = frames(stitched, N - 6, 12)
steps = [float(np.abs(around[i + 1] - around[i]).mean()) for i in range(len(around) - 1)]
print("stitched frame-to-frame mean abs diff, frames %d..%d (the cut is between %d and %d):" % (N - 6, N + 5, N - 1, N))
print("   " + " ".join("%.2f" % x for x in steps))
# 3) audio around the seam: RMS in 50 ms windows
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", stitched, "-ac", "1", "-ar", "16000", "/tmp/seam.wav"], check=True)
w = wave.open("/tmp/seam.wav")
a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
c = int(N / fps * 16000)
win = 800
rms = [float(np.sqrt(np.mean(a[c + k * win:c + (k + 1) * win] ** 2))) for k in range(-6, 6)]
print("audio RMS in 50 ms windows from -300 ms to +300 ms around the cut: " + " ".join("%.3f" % r for r in rms))
jump = float(np.abs(a[c] - a[c - 1]))
print("sample jump at the cut: %.4f (typical |diff| nearby: %.4f)" % (jump, float(np.mean(np.abs(np.diff(a[c - 1600:c + 1600]))))))
# seam frames for a visual check
import os
os.makedirs("/data/out/ltx/seam", exist_ok=True)
for k, i in ((0, N - 2), (1, N - 1), (2, N), (3, N + 1), (4, N + 24 * 3)):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", stitched, "-vf", "select=eq(n\\,%d),scale=640:-2" % i, "-vsync", "0",
                    "-frames:v", "1", "/data/out/ltx/seam/seam_%d_f%d.png" % (k, i)], check=True)
print("seam PNGs in /data/out/ltx/seam")
