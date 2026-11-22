"""OpenAI-style async video API (POST /v1/videos, GET /v1/videos/{id}, GET /v1/videos/{id}/content) in front of a
local ComfyUI running LTX-2.5 distilled (GGUF DiT) on the Halo's RTX 4090. Built 2026-11-22 for the mh gateway.

- Create returns at once with status "queued"; one GPU job runs at a time; clients poll, then download the mp4.
- Ids are returned already in LiteLLM's encoded form (video_ + base64 "litellm:custom_llm_provider:openai;
  model_id:<model>;video_id:<raw>", <model> = the `model` field LiteLLM sends), so status/content calls route back
  to this deployment; raw and encoded ids are both accepted.
- LiteLLM 1.98 forwards extra top-level fields (seed, negative_prompt, fps); <ltx_args>{"seed": 1}</ltx_args> in the
  prompt is a fallback for clients that cannot add fields (the tag is stripped).
- LiteLLM never checks the HTTP status of status/content GETs: unknown ids get a VideoObject-shaped "failed" body,
  and /content drops the connection for LiteLLM callers when there is no mp4 (else the error JSON becomes a .mp4).
- Jobs persist in LTX_STATE. run.sh restarts the whole container when ComfyUI or the adapter exits, so after a
  restart queued and in-flight jobs are re-queued and rendered from the start. ComfyUI prompt ids derive from the job
  id, which only guards against submitting a job twice to a ComfyUI that outlived the adapter.
- Self-heal: a job that times out, a ComfyUI that stops answering, or a sticky CUDA error fails the job, then the
  adapter exits so Docker restarts the container (and with it the CUDA context).
- Auth: "Authorization: Bearer <LTX_ENGINE_KEY>" on everything but GET /health; refuses to start without a key.
"""
import asyncio, base64, copy, hashlib, hmac, io, json, math, os, re, time, uuid
from pathlib import Path

from aiohttp import BodyPartReader, ClientError, ClientSession, ClientTimeout, web
from PIL import Image, ImageOps

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
MODEL_ID = os.environ.get("LTX_MODEL_ID", "ltx-2.5")
KEY = os.environ.get("LTX_ENGINE_KEY", "")
WF_DIR = Path(os.environ.get("LTX_WORKFLOWS", "/opt/adapter"))
OUT_DIR = Path(os.environ.get("COMFY_OUTPUT", "/data/out"))
IN_DIR = Path(os.environ.get("COMFY_INPUT", "/data/input"))
STATE = Path(os.environ.get("LTX_STATE", "/data/adapter/jobs.json"))
MAX_PIXELS = int(os.environ.get("LTX_MAX_PIXELS", str(1280 * 704)))
MAX_FRAMES = int(os.environ.get("LTX_MAX_FRAMES", "241"))
DEFAULT_SIZE = os.environ.get("LTX_DEFAULT_SIZE", "1280x704")
DEFAULT_SECONDS = float(os.environ.get("LTX_DEFAULT_SECONDS", "5"))
MAX_QUEUE = int(os.environ.get("LTX_MAX_QUEUE", "3"))
JOB_TIMEOUT = float(os.environ.get("LTX_JOB_TIMEOUT_S", "600"))
KEEP_HOURS = float(os.environ.get("LTX_KEEP_HOURS", "24"))
KEEP_BYTES = int(float(os.environ.get("LTX_KEEP_GB", "10")) * 1e9)
MAX_PROMPT = int(os.environ.get("LTX_MAX_PROMPT_CHARS", "8000"))
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PX = 4096 * 4096
DEFAULT_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"
HEALTH_PROMPT = "test video generation"  # LiteLLM's health-check probe for video_generation models
ALLOWED = {"prompt", "size", "seconds", "seed", "negative_prompt", "fps", "model", "user"}
MODEL_RE = re.compile(r"[A-Za-z0-9._:-]{1,64}")
RAW_RE = re.compile(r"ltx[0-9a-f]{32}")
STICKY_CUDA = re.compile(r"CUDA (error|driver)|cudaError|CUBLAS_STATUS|device-side assert")  # OOM is not sticky
STARTED = int(time.time())
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PX

if not KEY and os.environ.get("LTX_ALLOW_NO_AUTH") != "1":
    raise SystemExit("LTX_ENGINE_KEY is not set; refusing to start without auth")

T2V = json.loads((WF_DIR / "ltx25_distilled_gguf_t2v_api.json").read_text())
I2V = json.loads((WF_DIR / "ltx25_distilled_gguf_i2v_api.json").read_text())
jobs: dict[str, dict] = {}
recent: dict[str, tuple[float, str]] = {}  # payload hash -> (time, raw): absorbs LiteLLM's create retries
queue: asyncio.Queue = asyncio.Queue()


class BadRequest(Exception):
    pass


def encode_id(raw: str, model: str | None = None) -> str:
    s = "litellm:custom_llm_provider:openai;model_id:%s;video_id:%s" % (model or MODEL_ID, raw)
    return "video_" + base64.b64encode(s.encode()).decode()


def url_safe_model(model) -> bool:
    # hex raw ids never base64 to '/' or '+', so one probe decides for every id under this model name
    if not (isinstance(model, str) and MODEL_RE.fullmatch(model)):
        return False
    probe = encode_id("ltx" + "0" * 32, model)
    return "/" not in probe and "+" not in probe


if not url_safe_model(MODEL_ID):
    raise SystemExit("LTX_MODEL_ID %r is not usable in URL paths; pick a name of [A-Za-z0-9._-]" % MODEL_ID)


def raw_of(vid: str) -> str:
    if vid.startswith("video_"):
        try:
            s = vid[len("video_"):]
            s += "=" * (-len(s) % 4)
            m = re.search(r"video_id:([^;]+)$", base64.b64decode(s).decode())
            if m:
                return m.group(1)
        except Exception:
            pass
    return vid


def save_jobs(strict: bool = False):
    """Atomically write the job table. Only create() needs the write to succeed (strict); elsewhere a failed write
    (a full disk) is logged and the in-memory state carries on, so the next successful save catches up."""
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(jobs))
        os.replace(tmp, STATE)
    except OSError as e:
        if strict:
            raise
        print("[ltx-adapter] could not save %s: %r" % (STATE, e), flush=True)


def load_jobs():
    try:
        jobs.update(json.loads(STATE.read_text()))
    except (FileNotFoundError, ValueError):
        return
    for raw, j in sorted(jobs.items(), key=lambda kv: kv[1]["created_at"]):
        if j["status"] in ("queued", "in_progress"):
            j["status"], j["progress"] = "queued", 0
            queue.put_nowait(raw)


def lookup(vid: str):
    raw = raw_of(vid)
    job = jobs.get(raw)
    if job is None and RAW_RE.fullmatch(raw):  # finished before a lost state file: recover from disk
        hits = sorted((OUT_DIR / "ltx").glob(raw + "_*.mp4"))
        if hits:
            t = int(hits[-1].stat().st_mtime)
            job = jobs[raw] = {"raw": raw, "status": "completed", "progress": 100, "created_at": t, "completed_at": t,
                               "seconds": 0, "w": 0, "h": 0, "path": str(hits[-1]), "image": None}
    return raw, job


def gone(raw: str, model=None) -> dict:
    return {"id": encode_id(raw, model if url_safe_model(model) else None), "object": "video", "model": MODEL_ID,
            "status": "failed", "progress": 0, "created_at": 0, "seconds": "0", "size": "0x0",
            "error": {"code": "video_not_found", "message": "unknown or expired video id"}}


def parse_size(size) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d{1,5})\s*[xX*]\s*(\d{1,5})\s*", str(size or DEFAULT_SIZE))
    if not m:
        raise BadRequest("size must look like 1280x704")
    w, h = int(m.group(1)) // 64 * 64, int(m.group(2)) // 64 * 64  # two-stage pipeline: multiples of 64
    if w < 256 or h < 256:
        raise BadRequest("size too small (min 256 per side)")
    if max(w, h) > 3 * min(w, h):
        raise BadRequest("aspect ratio must be within 1:3")
    if w * h > MAX_PIXELS:  # scale down, keep aspect
        f = (MAX_PIXELS / (w * h)) ** 0.5
        w, h = int(w * f) // 64 * 64, int(h * f) // 64 * 64
    while w * h > MAX_PIXELS:
        if w >= h:
            w -= 64
        else:
            h -= 64
    if min(w, h) < 256:
        raise BadRequest("size cannot fit the pixel budget")
    return w, h


def frames_for(seconds, fps: int) -> int:
    try:
        sec = float(seconds) if seconds not in (None, "") else DEFAULT_SECONDS
    except (TypeError, ValueError):
        raise BadRequest("seconds must be a number")
    if not math.isfinite(sec) or not 0 < sec <= 60:
        raise BadRequest("seconds must be in (0, 60]")
    k = max(1, round(sec * fps / 8))
    return min(8 * k + 1, MAX_FRAMES // 8 * 8 + 1)


def sanitize_image(data: bytes) -> bytes:
    """Re-encode to a single-frame PNG: LoadImage would otherwise decode every frame of any ffmpeg-readable file."""
    if not data:
        raise BadRequest("input_reference is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise BadRequest("input_reference larger than 20 MB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ("PNG", "JPEG", "WEBP"):
                raise BadRequest("input_reference must be a PNG, JPEG or WebP image")
            if im.width * im.height > MAX_IMAGE_PX:
                raise BadRequest("input_reference larger than 4096x4096")
            im.seek(0)
            rgb = ImageOps.exif_transpose(im).convert("RGB")
    except BadRequest:
        raise
    except Exception as e:  # UnidentifiedImageError, DecompressionBombError, truncated files
        raise BadRequest("input_reference is not a readable image (%s)" % type(e).__name__)
    rgb.thumbnail((2048, 2048))
    buf = io.BytesIO()
    rgb.save(buf, "PNG")
    return buf.getvalue()


def public(job: dict) -> dict:
    model = job.get("model_id") or MODEL_ID
    o = {"id": encode_id(job["raw"], model), "object": "video", "model": model, "status": job["status"],
         "progress": job["progress"], "created_at": job["created_at"], "seconds": "%g" % job["seconds"],
         "size": "%dx%d" % (job["w"], job["h"])}
    if job.get("completed_at"):
        o["completed_at"] = job["completed_at"]
    if job["status"] == "completed":
        o["expires_at"] = job["completed_at"] + int(KEEP_HOURS * 3600)
        if not (job.get("path") and os.path.exists(job["path"])):
            o["status"], o["error"] = "failed", {"code": "video_expired", "message": "the output was deleted"}
    elif job["status"] == "expired":
        o["status"], o["error"] = "failed", {"code": "video_expired", "message": "the output expired"}
    elif job.get("error"):
        o["error"] = {"code": "generation_failed", "message": job["error"]}
    return o


def authorized(request) -> bool:
    if not KEY:
        return True
    return hmac.compare_digest(request.headers.get("Authorization", "").encode(), ("Bearer " + KEY).encode())


def unauthorized():
    return web.json_response({"error": {"message": "unauthorized", "type": "authentication_error"}}, status=401)


def waiting() -> int:
    return sum(1 for j in jobs.values() if j["status"] in ("queued", "in_progress"))


def queue_full():
    return web.json_response({"error": {"message": "video queue is full, retry later", "type": "rate_limit_error"}},
                             status=429, headers={"Retry-After": "30"})


async def read_capped(part, cap: int) -> bytes:
    buf = bytearray()
    while chunk := await part.read_chunk(256 * 1024):
        buf += chunk
        if len(buf) > cap:
            raise BadRequest("field %s too large" % part.name)
    return bytes(buf)


async def create(request):
    if not authorized(request):
        return unauthorized()
    fields, image_bytes = {}, None
    try:
        if request.content_type.startswith("multipart/"):
            async for part in await request.multipart():
                if not isinstance(part, BodyPartReader):
                    raise BadRequest("nested multipart is not supported")
                if part.name == "input_reference":
                    image_bytes = await read_capped(part, MAX_IMAGE_BYTES)
                elif part.name in ALLOWED:
                    fields[part.name] = (await read_capped(part, 64 * 1024)).decode("utf-8", "replace")
                else:
                    await part.release()
        elif request.content_type == "application/x-www-form-urlencoded":
            # LiteLLM's reconnect retry re-sends a multipart create without its file part: make the router retry it
            return web.json_response({"error": {"message": "multipart body lost in an upstream reconnect; retry",
                                                "type": "server_error"}}, status=503)
        else:
            fields = await request.json()
            if not isinstance(fields, dict):
                raise BadRequest("body must be a JSON object")
            ref = fields.get("input_reference")
            if isinstance(ref, str) and ref.startswith("data:"):
                image_bytes = ref.encode()
            elif ref is not None:
                raise BadRequest("input_reference must be an uploaded file or a data: URL")
        if image_bytes is not None and image_bytes[:5] == b"data:":  # through LiteLLM a JSON data: URL arrives as file bytes
            try:
                image_bytes = base64.b64decode(image_bytes.split(b",", 1)[1])
            except Exception:
                raise BadRequest("input_reference data URL is not valid base64")
        prompt = str(fields.get("prompt") or "")
        if len(prompt) > MAX_PROMPT + 4096:
            raise BadRequest("prompt too long (max %d chars)" % MAX_PROMPT)
        extra = {}
        i = prompt.find("<ltx_args>")
        if i >= 0:
            j = prompt.find("</ltx_args>", i)
            if j < 0:
                raise BadRequest("unterminated <ltx_args>")
            extra = json.loads(prompt[i + 10:j])
            if not isinstance(extra, dict):
                raise BadRequest("<ltx_args> must be a JSON object")
            prompt = prompt[:i] + prompt[j + 11:]
        prompt = prompt.strip()
        extra.update({k: fields[k] for k in ("seed", "negative_prompt", "fps") if fields.get(k) not in (None, "")})
        if not prompt:
            raise BadRequest("prompt is required")
        if len(prompt) > MAX_PROMPT:
            raise BadRequest("prompt too long (max %d chars)" % MAX_PROMPT)
        neg = str(extra.get("negative_prompt") or DEFAULT_NEGATIVE)
        if len(neg) > 2000:
            raise BadRequest("negative_prompt too long (max 2000 chars)")
        try:
            fps = int(extra.get("fps", 24))
            seed = int(extra["seed"]) if extra.get("seed") is not None else int.from_bytes(os.urandom(6), "big")
        except (TypeError, ValueError, OverflowError):
            raise BadRequest("fps and seed must be integers")
        if not 8 <= fps <= 50:
            raise BadRequest("fps must be 8..50")
        if not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise BadRequest("seed must be in 0..2^64-1")
        w, h = parse_size(fields.get("size"))
        frames = frames_for(fields.get("seconds"), fps)
        model = fields.get("model")
        model = model if url_safe_model(model) else MODEL_ID
        if prompt == HEALTH_PROMPT and image_bytes is None:  # answer LiteLLM's health probe without rendering
            alive = (await health(request)).status == 200
            o = {"id": encode_id("ltx" + "0" * 32, model), "object": "video", "model": model,
                 "status": "completed" if alive else "failed", "progress": 100 if alive else 0,
                 "created_at": int(time.time()), "seconds": "0", "size": "%dx%d" % (w, h)}
            if not alive:
                o["error"] = {"code": "unavailable", "message": "ComfyUI is down"}
            return web.json_response(o, status=200 if alive else 503)
        png = await asyncio.to_thread(sanitize_image, image_bytes) if image_bytes is not None else None
    except BadRequest as e:
        return web.json_response({"error": {"message": str(e), "type": "invalid_request_error"}}, status=400)
    except (ValueError, TypeError, OverflowError, AttributeError) as e:
        return web.json_response({"error": {"message": "bad request: %s" % e, "type": "invalid_request_error"}}, status=400)
    key = hashlib.sha256(json.dumps([prompt, neg, seed if extra.get("seed") is not None else None, fps, w, h, frames, model]).encode()
                         + (png or b"")).hexdigest()
    hit = recent.get(key)
    if hit and time.time() - hit[0] < 30 and hit[1] in jobs:
        return web.json_response(public(jobs[hit[1]]))
    if waiting() >= MAX_QUEUE:
        return queue_full()
    raw = "ltx" + uuid.uuid4().hex
    job = {"raw": raw, "status": "queued", "progress": 0, "created_at": int(time.time()), "prompt": prompt,
           "negative": neg, "seed": seed, "fps": fps, "frames": frames, "seconds": round((frames - 1) / fps, 2),
           "w": w, "h": h, "image": None, "model_id": model}
    try:
        if png is not None:
            IN_DIR.mkdir(parents=True, exist_ok=True)
            job["image"] = raw + ".png"
            (IN_DIR / job["image"]).write_bytes(png)
        jobs[raw] = job
        save_jobs(strict=True)
    except OSError as e:  # e.g. a full disk: leave no half-created job behind, and let the client retry
        jobs.pop(raw, None)
        if job["image"]:
            (IN_DIR / job["image"]).unlink(missing_ok=True)
        print("[ltx-adapter] cannot persist a new job: %r" % e, flush=True)
        return web.json_response({"error": {"message": "cannot store the job (%s); retry later" % type(e).__name__,
                                            "type": "server_error"}}, status=503)
    recent[key] = (time.time(), raw)
    queue.put_nowait(raw)
    return web.json_response(public(job))


async def status(request):
    if not authorized(request):
        return unauthorized()
    raw, job = lookup(request.match_info["vid"])
    if not job:  # LiteLLM ignores the status code and validates the body as a VideoObject
        return web.json_response(gone(raw, request.query.get("model")), status=404)
    return web.json_response(public(job))


async def content(request):
    from_litellm = request.headers.get("User-Agent", "").startswith("litellm/")
    if not authorized(request):
        if from_litellm and request.transport:
            request.transport.abort()
        return unauthorized()
    raw, job = lookup(request.match_info["vid"])
    if not job or job["status"] != "completed" or not job.get("path") or not Path(job["path"]).exists():
        # LiteLLM 1.98 relabels any /content body as 200 video/mp4; dropping the connection makes it raise instead
        if from_litellm and request.transport:
            request.transport.abort()
        if job and job["status"] in ("queued", "in_progress"):
            return web.json_response({"error": {"message": "video is %s" % job["status"]}}, status=409)
        return web.json_response({"error": {"message": "video not found or expired"}}, status=404)
    return web.FileResponse(job["path"], headers={"Content-Type": "video/mp4"})


async def models(request):
    if not authorized(request):
        return unauthorized()
    return web.json_response({"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]})


async def health(request):
    try:
        async with ClientSession(timeout=ClientTimeout(total=5)) as s, s.get(COMFY + "/system_stats") as r:
            ok = r.status == 200
    except Exception:
        ok = False
    w = request.app.get("worker")
    worker_ok = bool(w) and not w.done()
    good = ok and worker_ok
    return web.json_response({"ok": good, "comfyui": ok, "worker": worker_ok, "jobs_waiting": waiting(), "started": STARTED},
                             status=200 if good else 503)


def build_graph(job: dict) -> dict:
    g = copy.deepcopy(I2V if job["image"] else T2V)
    g["372"]["inputs"]["value"] = job["w"]
    g["360"]["inputs"]["value"] = job["h"]
    g["362"]["inputs"]["value"] = job["frames"]
    g["361"]["inputs"]["value"] = job["fps"]
    g["339"]["inputs"]["noise_seed"] = job["seed"]
    g["364"]["inputs"]["text"] = job["prompt"]
    g["373"]["inputs"]["text"] = job["negative"]
    g["75"]["inputs"]["filename_prefix"] = "ltx/" + job["raw"]
    if job["image"]:
        g["395"]["inputs"]["image"] = job["image"]
    return g


async def comfy_json(s: ClientSession, method: str, path: str, body=None):
    async with s.request(method, COMFY + path, json=body) as r:
        return r.status, await r.json(content_type=None)


async def comfy_queue_has(s: ClientSession, pid: str) -> bool:
    _, q = await comfy_json(s, "GET", "/queue")
    return any(len(it) > 1 and it[1] == pid for it in q.get("queue_running", []) + q.get("queue_pending", []))


async def comfy_cancel(s: ClientSession, pid: str) -> bool:
    for path, body in (("/queue", {"delete": [pid]}), ("/interrupt", {"prompt_id": pid})):
        try:
            await comfy_json(s, "POST", path, body)
        except Exception as e:  # noqa: BLE001
            print("[ltx-adapter] cancel %s failed: %r" % (path, e), flush=True)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if not await comfy_queue_has(s, pid):
                return True
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(2)
    return False


async def wait_comfy(s: ClientSession, seconds: float = 120) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            async with s.get(COMFY + "/system_stats") as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(3)
    return False


def harvest(entry: dict) -> str | None:
    for out in entry.get("outputs", {}).values():
        for v in out.values():
            for f in (v if isinstance(v, list) else []):
                if isinstance(f, dict) and str(f.get("filename", "")).endswith(".mp4"):
                    return str(OUT_DIR / f.get("subfolder", "") / f["filename"])
    return None


async def run_job(s: ClientSession, job: dict):
    job["status"], job["started"], job["progress"] = "in_progress", time.time(), 0
    save_jobs()
    pid = str(uuid.UUID(job["raw"][3:]))  # fixed per job: never submitted twice to the same ComfyUI
    _, h = await comfy_json(s, "GET", "/history/" + pid)
    if pid not in h and not await comfy_queue_has(s, pid):
        st, body = await comfy_json(s, "POST", "/prompt", {"prompt": build_graph(job), "client_id": job["raw"], "prompt_id": pid})
        if st != 200 or body.get("prompt_id") != pid:
            print("[ltx-adapter] %s ComfyUI rejected the graph: %s" % (job["raw"], json.dumps(body)[:2000]), flush=True)
            raise RuntimeError("generation failed (graph rejected)")
    expected = 25 + 0.33 * job["frames"] * job["w"] * job["h"] / (1280 * 704)  # rough seconds, for progress only
    polls = 0
    while True:
        await asyncio.sleep(2)
        polls += 1
        elapsed = time.time() - job["started"]
        job["progress"] = min(95, int(100 * elapsed / max(expected, 1)))
        if elapsed > JOB_TIMEOUT:  # ~6x the slowest normal render: something is stuck, so restart after failing it
            released = await comfy_cancel(s, pid)
            job["wedged"] = "a job timed out" + ("" if released else " and ComfyUI did not release it")
            raise RuntimeError("generation timed out after %d s" % JOB_TIMEOUT)
        try:
            _, h = await comfy_json(s, "GET", "/history/" + pid)
            if pid not in h:
                if polls % 15 == 0 and not await comfy_queue_has(s, pid):
                    _, h = await comfy_json(s, "GET", "/history/" + pid)  # it may have finished in between
                    if pid not in h:
                        raise RuntimeError("generation failed (ComfyUI lost the job)")
                if pid not in h:
                    continue
        except (ClientError, asyncio.TimeoutError, ValueError) as e:
            print("[ltx-adapter] %s poll error %r (retrying)" % (job["raw"], e), flush=True)
            continue
        entry = h[pid]
        st = entry.get("status", {})
        if st.get("status_str") == "error":
            err = next((m[1] for m in st.get("messages", []) if m[0] == "execution_error"), {})
            print("[ltx-adapter] %s ComfyUI error: %s" % (job["raw"], json.dumps(err)[:2000]), flush=True)
            if STICKY_CUDA.search(str(err.get("exception_message", ""))):
                job["wedged"] = "a sticky CUDA error"
            raise RuntimeError("generation failed (%s)" % (err.get("node_type") or "execution error"))
        path = harvest(entry)
        if path:
            job["path"] = path
            try:
                await comfy_json(s, "POST", "/history", {"delete": [pid]})
            except Exception:  # noqa: BLE001
                pass
            return
        if st.get("completed"):
            raise RuntimeError("generation failed (no video output)")


async def worker(app):
    async with ClientSession(timeout=ClientTimeout(total=60)) as s:
        while True:
            raw = await queue.get()
            job = jobs.get(raw)
            if not job or job["status"] != "queued":
                continue
            try:
                if not await wait_comfy(s):
                    job["wedged"] = "ComfyUI stopped answering"
                    raise RuntimeError("generation failed (ComfyUI is down)")
                await run_job(s, job)
                job["status"], job["progress"] = "completed", 100
            except Exception as e:  # noqa: BLE001
                job["status"], job["error"] = "failed", str(e) or type(e).__name__
            job["completed_at"] = int(time.time())
            save_jobs()
            print("[ltx-adapter] %s %s %dx%d %df in %.0f s %s" % (raw, job["status"], job["w"], job["h"], job["frames"],
                  max(0, time.time() - job.get("started", time.time())), job.get("error") or ""), flush=True)
            why = job.pop("wedged", None)
            if why:
                print("[ltx-adapter] %s; exiting so the container restarts" % why, flush=True)
                save_jobs()
                os._exit(1)


def expire(job: dict):
    for p in (job.get("path"), str(IN_DIR / job["image"]) if job.get("image") else None):
        if p:
            Path(p).unlink(missing_ok=True)
    job["status"] = "expired"


def janitor_pass():
    now = time.time()
    changed = False
    for raw, job in list(jobs.items()):
        done = job.get("completed_at")
        if not done:
            continue
        if job["status"] == "completed" and now - done > KEEP_HOURS * 3600:
            expire(job)
            changed = True
        elif job["status"] == "failed" and job.get("image") and now - done > KEEP_HOURS * 3600:
            (IN_DIR / job["image"]).unlink(missing_ok=True)
            job["image"] = None
            changed = True
        if now - done > 7 * 86400:  # forget the record a week after it finished
            jobs.pop(raw, None)
            changed = True
    by_path = {j.get("path"): j for j in jobs.values() if j.get("path")}
    files = []
    for p in (OUT_DIR / "ltx").glob("*.mp4"):
        try:
            st = p.stat()
            files.append((st.st_mtime, st.st_size, p))
        except OSError:
            pass
    files.sort()
    total = sum(f[1] for f in files)
    for mtime, size, p in files:
        if total <= KEEP_BYTES:
            break
        if now - mtime < 3600:
            continue
        j = by_path.get(str(p))
        if j:
            expire(j)
            changed = True
        else:
            p.unlink(missing_ok=True)
        total -= size
    for p in list(IN_DIR.glob("ltx*.png")) + list((OUT_DIR / "ltx").glob("ltx*.mp4")):
        stem = p.stem.split("_")[0]
        try:
            if stem not in jobs and now - p.stat().st_mtime > KEEP_HOURS * 3600:
                p.unlink(missing_ok=True)
        except OSError:
            pass
    for k, (t, _) in list(recent.items()):
        if now - t > 60:
            recent.pop(k, None)
    if changed:
        save_jobs()


async def janitor(app):
    while True:
        await asyncio.sleep(600)
        try:
            janitor_pass()
        except Exception as e:  # noqa: BLE001
            print("[ltx-adapter] janitor error: %r" % e, flush=True)


async def start_background(app):
    load_jobs()
    print("[ltx-adapter] model %s, %d jobs restored, %d re-queued" % (MODEL_ID, len(jobs), queue.qsize()), flush=True)
    app["worker"] = asyncio.create_task(worker(app))
    app["janitor"] = asyncio.create_task(janitor(app))

    def _die(t):
        if not t.cancelled() and t.exception():
            print("[ltx-adapter] worker died: %r" % t.exception(), flush=True)
            os._exit(1)

    app["worker"].add_done_callback(_die)


def main():
    app = web.Application(client_max_size=25 * 1024 * 1024)
    for prefix in ("/v1", ""):
        app.router.add_post(prefix + "/videos", create)
        app.router.add_get(prefix + "/videos/{vid}", status)
        app.router.add_get(prefix + "/videos/{vid}/content", content)
        app.router.add_get(prefix + "/models", models)
    app.router.add_get("/health", health)
    app.on_startup.append(start_background)
    web.run_app(app, host=os.environ.get("LTX_ADAPTER_HOST", "0.0.0.0"),
                port=int(os.environ.get("LTX_ADAPTER_PORT", "8101")), access_log=None)


if __name__ == "__main__":
    main()
