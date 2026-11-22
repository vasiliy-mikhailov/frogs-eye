"""Submit the LTX-2.5 T2V API workflow to ComfyUI (localhost:8188 via --network container:ltx) and time it."""
import json, os, sys, time, urllib.error, urllib.request, uuid

W, H, FRAMES, FPS, SEED = (int(x) for x in sys.argv[1:6])
PROMPT = os.environ.get("LTX_PROMPT") or sys.argv[6]
g = json.load(open("/b/adapter/ltx25_distilled_gguf_t2v_api.json"))
g["372"]["inputs"]["value"] = W
g["360"]["inputs"]["value"] = H
g["362"]["inputs"]["value"] = FRAMES
g["361"]["inputs"]["value"] = FPS
g["339"]["inputs"]["noise_seed"] = SEED
g["364"]["inputs"]["text"] = PROMPT
g["75"]["inputs"]["filename_prefix"] = "ltx/test_%dx%d_%df" % (W, H, FRAMES)
B = "http://127.0.0.1:8188"


def call(path, data=None):
    req = urllib.request.Request(B + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


t0 = time.time()
try:
    pid = call("/prompt", {"prompt": g, "client_id": str(uuid.uuid4())})["prompt_id"]
except urllib.error.HTTPError as e:
    print("SUBMIT ERROR", e.code, e.read().decode()[:2000]); sys.exit(1)
print("submitted", pid, flush=True)
last = ""
while True:
    time.sleep(5)
    h = call("/history/" + pid)
    if pid in h:
        st = h[pid].get("status", {})
        msgs = [(m[0], round(m[1].get("timestamp", 0) / 1000 - t0, 1)) for m in st.get("messages", [])]
        print("status:", st.get("status_str"), "events:", msgs, flush=True)
        if st.get("status_str") == "error":
            for m in st.get("messages", []):
                if m[0] == "execution_error":
                    print("ERROR node", m[1].get("node_type"), ":", m[1].get("exception_message", "")[:1500])
            sys.exit(1)
        for nid, out in h[pid].get("outputs", {}).items():
            for k, v in out.items():
                if isinstance(v, list):
                    for f in v:
                        if isinstance(f, dict) and "filename" in f:
                            print("OUTPUT", f.get("subfolder", ""), f["filename"])
        print("TOTAL %.1f s" % (time.time() - t0))
        break
    q = call("/queue")
    s = "running" if q.get("queue_running") else "pending"
    if s != last:
        print("%.0fs %s" % (time.time() - t0, s), flush=True); last = s
