#!/usr/bin/env python3
"""Generate a video clip (with sound) with LTX-2.5 through the gateway, wait for it, and save the mp4.

  python3 ltx_video.py --prompt "..." [--image first_frame.png] [--seconds 5] [--size 1280x704]
                       [--seed N] [--negative "..."] [--fps 24] [--out videos/name.mp4] [--max-wait 900]
  python3 ltx_video.py --resume <video id> [--out videos/name.mp4]      # pick up a job that is still running

Talks to https://inference.mikhailov.tech/ltx-2.5/v1 (OpenAI videos API; override with LTX_BASE_URL). The gateway key
comes from $LTX_API_KEY, $MIKHAILOV_API_KEY or $MIKHAILOV_JAVAFAKER_KEY, else from the `refs` in ~/.dsh/.credentials.yaml.
On the GPU box itself (the Strix Halo, where ~/.ltx/engine.env exists) it talks to the local adapter
(http://127.0.0.1:8101/v1) with that engine key instead, unless LTX_API_KEY or LTX_BASE_URL is set. The key is never
printed. Standard library only.

Exit codes: 0 saved; 1 generation or network failure; 2 bad request; 3 not finished within --max-wait (use --resume).
"""
import argparse, base64, json, os, random, re, struct, sys, time, urllib.error, urllib.request

BASE = os.environ.get("LTX_BASE_URL", "https://inference.mikhailov.tech/ltx-2.5/v1").rstrip("/")
MODEL = "ltx-2.5"
MAX_PIXELS = 1280 * 704
UA = "ltx-video-skill/1.0"


def api_key() -> str:
    """The gateway key, or on the GPU box itself the local adapter's key (and the local adapter as BASE)."""
    global BASE
    if os.environ.get("LTX_API_KEY", "").strip():
        return os.environ["LTX_API_KEY"].strip()
    if "LTX_BASE_URL" not in os.environ:
        # On the Strix Halo the adapter is local: going out to the gateway and back hairpins through the home router,
        # which moved a 0.6 MB clip in 59 s (2026-11-22).
        try:
            env = open(os.path.expanduser(os.path.join("~", ".ltx", "engine.env")), encoding="utf-8").read()
            m = re.search(r"^LTX_ENGINE_KEY=(\S+)", env, re.M)
            if m:
                BASE = "http://127.0.0.1:8101/v1"
                return m.group(1)
        except OSError:
            pass
    for name in ("MIKHAILOV_API_KEY", "MIKHAILOV_JAVAFAKER_KEY"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    home = os.environ.get("DSH_HOME") or os.path.expanduser("~/.dsh")
    try:
        text = open(os.path.join(home, ".credentials.yaml"), encoding="utf-8").read()
    except OSError:
        text = ""
    refs = re.search(r"^refs:[ \t]*\n((?:[ \t]+.*(?:\n|$))*)", text, re.M)
    if refs:
        for name in ("MIKHAILOV_API_KEY", "MIKHAILOV_JAVAFAKER_KEY"):
            m = re.search(r"^[ \t]+%s:[ \t]*['\"]?([^'\"\s#]+)" % name, refs.group(1), re.M)
            if m:
                return m.group(1)
    sys.exit("ltx_video: no gateway key found (set LTX_API_KEY, or MIKHAILOV_API_KEY in the dsh credentials refs)")


def call(method: str, path: str, key: str, body=None, raw=False, timeout=60):
    req = urllib.request.Request(BASE + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                          "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            return r.status, (data if raw else json.loads(data or b"{}")), r.headers
    except urllib.error.HTTPError as e:
        data = e.read()
        try:
            body = json.loads(data)
        except ValueError:
            body = {}
        if isinstance(body, dict) and body.get("object") == "video":
            return e.code, body, e.headers  # the adapter answers an unknown id with 404 + a "failed" video object
        msg = body.get("error", {}) if isinstance(body, dict) else {}
        msg = msg.get("message", msg) if isinstance(msg, dict) else msg
        inner = re.search(r'Exception - (\{.*\})', str(msg), re.S)  # LiteLLM wraps the adapter's own error JSON
        if inner:
            try:
                msg = json.loads(inner.group(1))["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
        return e.code, {"error": {"message": str(msg or data[:300].decode("utf-8", "replace"))}}, e.headers


def image_size(data: bytes):
    """(width, height) of a PNG, JPEG or WebP, or None."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
        return None
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        kind = data[12:16]
        if kind == b"VP8 " and len(data) >= 30:
            w, h = struct.unpack("<HH", data[26:30])
            return w & 0x3FFF, h & 0x3FFF
        if kind == b"VP8L" and len(data) >= 25:
            b = int.from_bytes(data[21:25], "little")
            return (b & 0x3FFF) + 1, ((b >> 14) & 0x3FFF) + 1
        if kind == b"VP8X" and len(data) >= 30:
            return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
    return None


def size_for(w0: int, h0: int) -> str:
    """The largest WxH (multiples of 64, within the pixel budget and 1:3) with the image's aspect ratio."""
    ratio = max(1 / 3, min(3, w0 / h0))
    best = (0, 0, 256, 256)  # (area, -aspect error, w, h)
    for h in range(256, 2049, 64):
        w = round(h * ratio / 64) * 64
        if w < 256 or w * h > MAX_PIXELS or max(w, h) > 3 * min(w, h):
            continue
        err = abs(w / h - ratio) / ratio
        if err <= 0.08 and (w * h, -err) > best[:2]:
            best = (w * h, -err, w, h)
    return "%dx%d" % best[2:]


def out_path(arg):
    path = arg or os.path.join("videos", time.strftime("ltx-%Y%m%d-%H%M%S.mp4"))
    if not path.endswith(".mp4"):
        path += ".mp4"
    base, n = path[:-4], 2
    while os.path.exists(path):
        path, n = "%s-%d.mp4" % (base, n), n + 1
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def main():
    ap = argparse.ArgumentParser(description="Generate a video with LTX-2.5 via inference.mikhailov.tech")
    ap.add_argument("--prompt", help="what happens in the clip (English works best)")
    ap.add_argument("--image", help="PNG/JPEG/WebP used as the first frame (image-to-video)")
    ap.add_argument("--seconds", type=float, default=5, help="clip length, up to 10 (default 5)")
    ap.add_argument("--size", help="WxH, e.g. 1280x704 (default), 704x1280, 896x896; default follows --image's aspect")
    ap.add_argument("--seed", type=int, help="fix for reproducible results (random if omitted, and printed)")
    ap.add_argument("--negative", help="negative prompt (things to avoid)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", help="output .mp4 path (default videos/ltx-<time>.mp4 under the current directory)")
    ap.add_argument("--max-wait", type=float, default=900, help="seconds to wait, queue time included (default 900)")
    ap.add_argument("--resume", metavar="VIDEO_ID", help="wait for and download an existing job instead of creating one")
    a = ap.parse_args()
    key = api_key()
    t0 = time.time()

    if a.resume:
        vid = a.resume
    else:
        if not a.prompt:
            ap.error("--prompt is required (or --resume)")
        seed = a.seed if a.seed is not None else random.randrange(1, 2 ** 48)
        body = {"model": MODEL, "prompt": a.prompt, "seconds": "%g" % a.seconds, "seed": seed, "fps": a.fps}
        if a.negative:
            body["negative_prompt"] = a.negative
        if a.image:
            data = open(a.image, "rb").read()
            if len(data) > 20 * 1024 * 1024:
                sys.exit("ltx_video: --image is larger than 20 MB")
            dims = image_size(data)
            mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg" if data[:2] == b"\xff\xd8" else "image/webp"
            body["input_reference"] = "data:%s;base64,%s" % (mime, base64.b64encode(data).decode())
            if not a.size and dims:
                a.size = size_for(*dims)
        body["size"] = a.size or "1280x704"
        while True:
            st, o, hd = call("POST", "/videos", key, body)
            if st == 429 and time.time() - t0 < a.max_wait:  # one GPU, short queue: wait our turn
                wait = min(60, int((hd.get("Retry-After") if hd else None) or 30))
                print("queue full, retrying in %d s" % wait, flush=True)
                time.sleep(wait)
                continue
            break
        if st != 200:
            print("create failed: HTTP %s: %s" % (st, o.get("error", {}).get("message")), flush=True)
            sys.exit(2 if st == 400 else 1)
        vid = o["id"]
        print("queued %s: %s, %s s, seed %d" % (vid[:24] + "...", o.get("size"), o.get("seconds"), seed), flush=True)

    last, last_print = None, 0
    while True:
        try:
            st, o, _ = call("GET", "/videos/" + vid, key, timeout=30)
        except (urllib.error.URLError, OSError) as e:  # a network blip or a restarting box: keep polling
            st, o = type(e).__name__, {}
        status = o.get("status") if st in (200, 404) else None
        if status in ("completed", "failed"):
            break
        now = time.time()
        if status != last or now - last_print > 30:
            print("%4.0f s  %s  %s%%" % (now - t0, status or "HTTP %s" % st, o.get("progress", "?")), flush=True)
            last, last_print = status, now
        if now - t0 > a.max_wait:
            print("not finished after %.0f s (status %s). Resume with:\n  python3 %s --resume %s%s"
                  % (now - t0, status, sys.argv[0], vid, " --out " + a.out if a.out else ""), flush=True)
            sys.exit(3)
        time.sleep(5)

    if status == "failed":
        err = o.get("error") or {}
        print("generation failed: %s (%s)" % (err.get("message"), err.get("code")), flush=True)
        sys.exit(1)
    path = out_path(a.out)
    st, data, _ = call("GET", "/videos/%s/content" % vid, key, raw=True, timeout=300)
    if st != 200 or data[4:8] != b"ftyp":
        print("download failed: HTTP %s" % st, flush=True)
        sys.exit(1)
    open(path, "wb").write(data)
    print("saved %s (%s s, %s, %.1f MB, %.0f s total)\nvideo id: %s"
          % (os.path.abspath(path), o.get("seconds"), o.get("size"), len(data) / 1e6, time.time() - t0, vid), flush=True)


if __name__ == "__main__":
    main()
