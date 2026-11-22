"""Public-gateway test of ltx-2.5 with the OpenAI SDK: T2V, an unknown id, a bad size, and (arg `i2v`) image-to-video.
Run it where the openai package and the LiteLLM master key are, e.g. inside the litellm container on mh:
  docker cp gateway_test.py litellm:/tmp/ && docker exec litellm python -u /tmp/gateway_test.py i2v
Never prints the key; clips land in /tmp/gw_{t2v,i2v}.mp4."""
import io, os, struct, sys, time, zlib

from openai import OpenAI

BASE = os.environ.get("LTX_GATEWAY_URL", "https://inference.mikhailov.tech/ltx-2.5/v1")
c = OpenAI(base_url=BASE, api_key=os.environ["LITELLM_MASTER_KEY"], timeout=120, max_retries=0)


def run(label, **kw):
    t0 = time.time()
    v = c.videos.create(model="ltx-2.5", **kw)
    print("%s create: %s %s %s (%.1f s)" % (label, v.status, v.size, v.seconds, time.time() - t0), flush=True)
    last = None
    while v.status in ("queued", "in_progress"):
        time.sleep(6)
        v = c.videos.retrieve(v.id)
        if v.status != last:
            print("   %4.0f s %s %s%%" % (time.time() - t0, v.status, v.progress), flush=True)
            last = v.status
    print("%s final: %s %s (%.0f s total)" % (label, v.status, v.error, time.time() - t0), flush=True)
    if v.status == "completed":
        data = c.videos.download_content(v.id).read()
        out = "/tmp/gw_%s.mp4" % label
        open(out, "wb").write(data)
        print("%s download: %d bytes, ftyp=%s -> %s" % (label, len(data), data[4:8] == b"ftyp", out), flush=True)
    return v


v = run("t2v", prompt="A paper boat drifts down a rain-filled street gutter, close-up, cinematic, shallow depth of field",
        size="1280x704", seconds="5")
# negative paths through the real gateway
try:
    c.videos.retrieve("video_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpsdHgtMi41O3ZpZGVvX2lkOmx0eGFiYWJhYmFiYWJhYmFiYWJhYmFiYWJhYmFiYWJhYmFi")
    print("unknown id: no exception")
except Exception as e:
    print("unknown id: %s" % type(e).__name__)
r = c.videos.retrieve("video_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpsdHgtMi41O3ZpZGVvX2lkOmx0eGFiYWJhYmFiYWJhYmFiYWJhYmFiYWJhYmFiYWJhYmFi")
print("unknown id ->", r.status, r.error)
try:
    c.videos.create(model="ltx-2.5", prompt="x", size="256x16384")
    print("bad size: accepted?!")
except Exception as e:
    print("bad size: %s %s" % (type(e).__name__, str(e)[:160]))
if len(sys.argv) > 1 and sys.argv[1] == "i2v":
    W, H = 1280, 704
    rows = bytearray()
    for y in range(H):
        rows.append(0)
        for x in range(W):
            ridge = y > 704 - (abs(x - 450) * -0.7 + 300) or y > 704 - (abs(x - 950) * -0.8 + 320)
            star = (x * 7919 + y * 104729) % 2003 == 0 and y < 450
            rows += bytes((60, 70, 60) if ridge else (255, 255, 230) if star else (20, 30, 70 + y // 20))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b"")
    buf = io.BytesIO(png); buf.name = "night.png"
    run("i2v", prompt="Stars twinkle over dark mountains, a slow camera push-in, a meteor streaks across the sky",
        size="1280x704", seconds="4", input_reference=buf)
