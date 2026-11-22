"""Local test of the LTX adapter (run inside --network container:ltx with the engine env). Never prints the key."""
import base64, io, json, os, sys, time, urllib.error, urllib.request, uuid

BASE = "http://127.0.0.1:8101/v1"
KEY = os.environ["LTX_ENGINE_KEY"]
AUTH = {"Authorization": "Bearer " + KEY}
results = []


def req(method, path, body=None, headers=None, ctype="application/json", raw=False):
    h = dict(headers or {})
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        h["Content-Type"] = ctype
    r = urllib.request.Request(BASE + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            b = resp.read()
            return resp.status, (b if raw else json.loads(b or b"{}")), dict(resp.headers)
    except urllib.error.HTTPError as e:
        b = e.read()
        try:
            return e.code, json.loads(b), dict(e.headers)
        except ValueError:
            return e.code, b[:200], dict(e.headers)


def check(name, cond, info=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + ("  " + str(info)[:300] if info else ""), flush=True)


def wait(vid, limit=400):
    t0 = time.time()
    last = None
    while time.time() - t0 < limit:
        st, o, _ = req("GET", "/videos/" + vid, headers=AUTH)
        if o.get("status") != last:
            print("   %5.0fs %s %s%%" % (time.time() - t0, o.get("status"), o.get("progress")), flush=True)
            last = o.get("status")
        if o.get("status") in ("completed", "failed"):
            return o, time.time() - t0
        time.sleep(3)
    return o, time.time() - t0


def health_started(limit=300):
    """Wait until the adapter answers /health with 200 (it is down while the container restarts); return its start time."""
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            with urllib.request.urlopen(BASE.replace("/v1", "") + "/health", timeout=5) as r:
                return json.loads(r.read())["started"]
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            time.sleep(3)
    return None


def multipart(fields, files):
    bd = "----ltx" + uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (bd, k, v)).encode())
    for k, (fn, data, ct) in files.items():
        out.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\nContent-Type: %s\r\n\r\n" % (bd, k, fn, ct)).encode())
        out.write(data + b"\r\n")
    out.write(("--%s--\r\n" % bd).encode())
    return out.getvalue(), "multipart/form-data; boundary=" + bd


phase = sys.argv[1] if len(sys.argv) > 1 else "main"

if phase == "main":
    # auth
    for m, p in (("POST", "/videos"), ("GET", "/videos/x"), ("GET", "/models")):
        st, o, _ = req(m, p, body={"prompt": "x"} if m == "POST" else None)
        check("no key -> 401 %s %s" % (m, p), st == 401, st)
    st, o, _ = req("GET", "/models", headers={"Authorization": "Bearer wrong"})
    check("wrong key -> 401", st == 401, st)
    st, o, _ = req("GET", "/models", headers=AUTH)
    check("models with key", st == 200 and o["data"][0]["id"] == "ltx-2.5", o)
    # bad input -> 400
    bad = [{"prompt": "x", "size": "300x99999999"}, {"prompt": "x", "size": "256x16384"}, {"prompt": "x", "seconds": "inf"},
           {"prompt": "x", "seconds": 1e300}, {"prompt": "x", "seed": "abc"}, {"prompt": "x", "seed": -1},
           {"prompt": "x <ltx_args>[1]</ltx_args>"}, {"prompt": "x <ltx_args>{\"seed\": null"},
           {"prompt": "<ltx_args>" * 100000}, {"prompt": ""}, {"prompt": "x", "fps": 200},
           {"prompt": "x", "input_reference": "data:image/png;base64,AAAA"}, {"prompt": "x", "input_reference": 5}]
    for b in bad:
        t0 = time.time()
        st, o, _ = req("POST", "/videos", b, AUTH)
        check("400 for %s" % json.dumps(b)[:60], st == 400 and time.time() - t0 < 2, (st, o.get("error", {}).get("message") if isinstance(o, dict) else o))
    st, o, _ = req("POST", "/videos", b"[1,2]", AUTH)
    check("400 for non-object body", st == 400, st)
    st, o, _ = req("POST", "/videos", b"prompt=x", AUTH, ctype="application/x-www-form-urlencoded")
    check("503 for urlencoded (lost multipart)", st == 503, st)
    body, ct = multipart({"prompt": "x"}, {"input_reference": ("input_reference.png", b"\x00\x00\x00\x18ftypmp42" + b"x" * 5000, "image/png")})
    st, o, _ = req("POST", "/videos", body, AUTH, ctype=ct)
    check("400 for an mp4 posing as png", st == 400, o)
    # health probe: no render
    st, o, _ = req("POST", "/videos", {"prompt": "test video generation", "model": "ltx-2.5"}, AUTH)
    check("health probe answered without a job", st == 200 and o["status"] == "completed", o)
    # unknown id -> VideoObject failed
    fake = "video_" + __import__("base64").b64encode(b"litellm:custom_llm_provider:openai;model_id:ltx-2.5;video_id:ltx" + b"ab" * 16).decode()
    st, o, _ = req("GET", "/videos/" + fake, headers=AUTH)
    check("unknown id -> 404 + failed VideoObject", st == 404 and o.get("status") == "failed" and o["error"]["code"] == "video_not_found", o)
    # a valid data: URL, sent directly as JSON and the way LiteLLM forwards it (the URL text as the file part)
    from PIL import Image
    small = io.BytesIO(); Image.new("RGB", (512, 512), (200, 120, 40)).save(small, "PNG")
    durl = "data:image/png;base64," + base64.b64encode(small.getvalue()).decode()
    dj = {"model": "ltx-2.5", "prompt": "An orange square slowly fades to black", "size": "512x512", "seconds": "1", "seed": 7}
    st, o, _ = req("POST", "/videos", dict(dj, input_reference=durl), AUTH)
    check("data: URL in JSON -> queued I2V", st == 200 and o.get("status") == "queued", (st, o))
    body, ct = multipart({k: str(v) for k, v in dj.items()}, {"input_reference": ("input_reference.png", durl.encode(), "image/png")})
    st, o2, _ = req("POST", "/videos", body, AUTH, ctype=ct)
    check("data: URL as multipart file bytes (LiteLLM's form) -> same job", st == 200 and o2.get("id") == o.get("id"), (st, o2.get("id") == o.get("id")))
    od, dt = wait(o["id"])
    check("data: URL I2V completed", od["status"] == "completed", (round(dt), od.get("status"), od.get("error")))
    # T2V
    t2v = {"model": "ltx-2.5", "prompt": "A red fox trots across a snowy meadow at sunrise, cinematic, soft light",
           "size": "768x512", "seconds": "3", "seed": 42}
    st, o, _ = req("POST", "/videos", t2v, AUTH)
    check("T2V create", st == 200 and o["status"] == "queued" and o["id"].startswith("video_") and "/" not in o["id"] and "+" not in o["id"], o)
    vid = o["id"]
    st, o2, _ = req("POST", "/videos", t2v, AUTH)
    check("identical retry within 30 s -> same job", o2.get("id") == vid, o2.get("id") == vid)
    # content before ready: direct -> 409, litellm UA -> connection dropped
    st, o3, _ = req("GET", "/videos/%s/content" % vid, headers=AUTH)
    check("content before ready (direct) -> 409", st == 409, st)
    try:
        st, o3, _ = req("GET", "/videos/%s/content" % vid, headers=dict(AUTH, **{"User-Agent": "litellm/1.98.0"}))
        check("content before ready (litellm UA) -> dropped", False, st)
    except Exception as e:
        check("content before ready (litellm UA) -> dropped", True, type(e).__name__)
    o, dt = wait(vid)
    check("T2V completed", o["status"] == "completed" and o.get("expires_at"), (round(dt), o))
    st, mp4, hd = req("GET", "/videos/%s/content" % vid, headers=AUTH, raw=True)
    check("T2V content is mp4", st == 200 and mp4[4:8] == b"ftyp" and len(mp4) > 10000, (st, len(mp4), hd.get("Content-Type")))
    check("no workflow metadata in mp4", b"UnetLoaderGGUF" not in mp4 and b"ltx-2.5-22b" not in mp4)
    open("/t/test_t2v.mp4", "wb").write(mp4)
    # extend: continue the T2V clip by 2 s (its 73 frames are shorter than the 5 s overlap, so all of them are pinned)
    st, o, _ = req("POST", "/videos", {"model": "ltx-2.5", "prompt": "x", "extend_from": "video_unknown"}, AUTH)
    check("extend_from an unknown id -> 400", st == 400, (st, o.get("error", {}).get("message")))
    st, o, _ = req("POST", "/videos", {"model": "ltx-2.5", "prompt": "The fox stops and looks back at the camera, wind in the grass",
                                       "extend_from": vid, "seconds": "2", "seed": 7}, AUTH)
    check("extend create", st == 200 and o.get("status") == "queued" and o.get("seconds") == "5", (st, o))
    oe, dt = wait(o["id"])
    check("extend completed", oe["status"] == "completed", (round(dt), oe.get("status"), oe.get("error")))
    st, mp4e, _ = req("GET", "/videos/%s/content" % o["id"], headers=AUTH, raw=True)
    open("/t/test_ext.mp4", "wb").write(mp4e)
    import subprocess
    pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,nb_frames,duration", "-of", "json",
                                    "/t/test_ext.mp4"], capture_output=True, text=True).stdout or "{}")
    vf = [x for x in pr.get("streams", []) if x.get("codec_type") == "video"]
    af = [x for x in pr.get("streams", []) if x.get("codec_type") == "audio"]
    check("extended mp4 = 73 + 48 frames, with sound as long as the picture",
          vf and vf[0].get("nb_frames") == "121" and af and abs(float(af[0]["duration"]) - 121 / 24) < 0.06, pr)
    # I2V with a generated PNG
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1024, 704), (30, 60, 120))
    d = ImageDraw.Draw(im)
    d.ellipse((380, 220, 640, 480), fill=(250, 200, 40))
    d.rectangle((0, 560, 1024, 704), fill=(40, 120, 50))
    buf = io.BytesIO(); im.save(buf, "PNG")
    body, ct = multipart({"model": "ltx-2.5", "prompt": "The sun slowly rises over a green field, clouds drift by", "size": "1024x704", "seconds": "3"},
                         {"input_reference": ("input_reference.png", buf.getvalue(), "image/png")})
    st, o, _ = req("POST", "/videos", body, AUTH, ctype=ct)
    check("I2V create", st == 200 and o["status"] == "queued", o)
    o, dt = wait(o["id"])
    check("I2V completed", o["status"] == "completed", (round(dt), o))
    st, mp4, _ = req("GET", "/videos/%s/content" % o["id"], headers=AUTH, raw=True)
    check("I2V content is mp4", st == 200 and mp4[4:8] == b"ftyp", len(mp4))
    open("/t/test_i2v.mp4", "wb").write(mp4)
    json.dump({"t2v": vid}, open("/t/test_ids.json", "w"))

elif phase == "restart-create":
    st, o, _ = req("POST", "/videos", {"model": "ltx-2.5", "prompt": "Waves crash on a rocky shore under a stormy sky", "size": "768x512", "seconds": "3"}, AUTH)
    check("create before restart", st == 200, o)
    ids = json.load(open("/t/test_ids.json")); ids["restart"] = o["id"]; ids["started"] = health_started()
    json.dump(ids, open("/t/test_ids.json", "w"))
    for _ in range(30):
        st, o, _ = req("GET", "/videos/" + ids["restart"], headers=AUTH)
        if o["status"] == "in_progress":
            break
        time.sleep(1)
    check("job in_progress before the kill", o["status"] == "in_progress", o["status"])
    print("now run: docker exec ltx pkill -f ltx_video_adapter.py   (then restart-check)")

elif phase == "restart-check":
    ids = json.load(open("/t/test_ids.json"))
    started = health_started()
    check("adapter came back with a new process (it restarted)", started is not None and started != ids.get("started"),
          (ids.get("started"), started))
    st, o, _ = req("GET", "/videos/" + ids["t2v"], headers=AUTH)
    check("completed job survives a restart", st == 200 and o["status"] == "completed", o)
    st, mp4, _ = req("GET", "/videos/%s/content" % ids["t2v"], headers=AUTH, raw=True)
    check("its mp4 is still served", st == 200 and mp4[4:8] == b"ftyp", (st, len(mp4)))
    o, dt = wait(ids["restart"])
    check("in-flight job re-queued and completed after restart", o["status"] == "completed", (round(dt), o))

print("SUMMARY %d/%d passed" % (sum(r[1] for r in results), len(results)))
