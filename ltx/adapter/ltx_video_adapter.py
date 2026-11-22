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
- Self-heal: a job that times out, a ComfyUI that stops answering, or a sticky CUDA error ends the job, then the
  adapter exits so Docker restarts the container (and with it the CUDA context). The last two are GPU faults (a
  Windows TDR resetting the eGPU was seen live), so their job is re-queued once instead of failed.
- Extend: `extend_from: <video id>` continues a finished video. The last LTX_EXTEND_OVERLAP_FRAMES (5 s) of it are
  pinned as the first frames and sound of a new render (the ext graph), the overlap is cut off the result and the rest
  is appended. The job's output is the whole video, which can be extended again, up to LTX_MAX_TOTAL_S.
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
MAX_FRAMES = int(os.environ.get("LTX_MAX_FRAMES", "481"))  # 20 s at 24 fps: 170 s and 21.6 GB at 1280x704 on the 4090
OVERLAP_FRAMES = int(os.environ.get("LTX_EXTEND_OVERLAP_FRAMES", "121"))  # 8k+1: 5 s at 24 fps
MAX_TOTAL_S = float(os.environ.get("LTX_MAX_TOTAL_S", "900"))  # longest video an extension may produce
XFADE_S = 0.1  # audio crossfade at an extension seam
DEFAULT_SIZE = os.environ.get("LTX_DEFAULT_SIZE", "1280x704")
DEFAULT_SECONDS = float(os.environ.get("LTX_DEFAULT_SECONDS", "5"))
MAX_QUEUE = int(os.environ.get("LTX_MAX_QUEUE", "3"))
JOB_TIMEOUT = float(os.environ.get("LTX_JOB_TIMEOUT_S", "600"))  # counted from when ComfyUI starts running the job
QUEUE_WAIT = float(os.environ.get("LTX_QUEUE_WAIT_S", "7200"))  # waiting behind other ComfyUI work (e.g. the editor)
KEEP_HOURS = float(os.environ.get("LTX_KEEP_HOURS", "24"))
KEEP_BYTES = int(float(os.environ.get("LTX_KEEP_GB", "30")) * 1e9)
MAX_PROMPT = int(os.environ.get("LTX_MAX_PROMPT_CHARS", "8000"))
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PX = 4096 * 4096
DEFAULT_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"
HEALTH_PROMPT = "test video generation"  # LiteLLM's health-check probe for video_generation models
ALLOWED = {"prompt", "size", "seconds", "seed", "negative_prompt", "fps", "model", "user", "extend_from"}
MODEL_RE = re.compile(r"[A-Za-z0-9._:-]{1,64}")
RAW_RE = re.compile(r"ltx[0-9a-f]{32}")
STICKY_CUDA = re.compile(r"CUDA (error|driver)|cudaError|CUBLAS_STATUS|device-side assert")  # OOM is not sticky
STARTED = int(time.time())
RETRY_REASONS = ("a sticky CUDA error", "ComfyUI stopped answering")  # GPU faults: a job is re-queued once
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PX

if not KEY and os.environ.get("LTX_ALLOW_NO_AUTH") != "1":
    raise SystemExit("LTX_ENGINE_KEY is not set; refusing to start without auth")

T2V = json.loads((WF_DIR / "ltx25_distilled_gguf_t2v_api.json").read_text())
I2V = json.loads((WF_DIR / "ltx25_distilled_gguf_i2v_api.json").read_text())
EXT = json.loads((WF_DIR / "ltx25_distilled_gguf_ext_api.json").read_text())
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
        extra.update({k: fields[k] for k in ("seed", "negative_prompt", "fps", "extend_from") if fields.get(k) not in (None, "")})
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
        ext = None
        if extra.get("extend_from") not in (None, ""):
            if image_bytes is not None:
                raise BadRequest("extend_from cannot be combined with input_reference")
            src_raw, src = lookup(str(extra["extend_from"]))
            if not src or src["status"] != "completed" or not src.get("path") or not os.path.exists(src["path"]):
                raise BadRequest("extend_from must name a completed video that has not expired")
            try:
                w, h, fps, src_frames = await source_meta(src)
            except (RuntimeError, OSError, ValueError, KeyError) as e:
                raise BadRequest("cannot read the source video (%s)" % type(e).__name__)
            overlap = min(OVERLAP_FRAMES, (src_frames - 1) // 8 * 8 + 1)
            max_new = (MAX_FRAMES - OVERLAP_FRAMES) // 8 * 8  # 15 s, also when a short source carries over whole
            sec = fields.get("seconds")
            try:
                sec = float(sec) if sec not in (None, "") else max_new / fps
            except (TypeError, ValueError):
                raise BadRequest("seconds must be a number")
            if not math.isfinite(sec) or not 0 < sec <= 60:
                raise BadRequest("seconds must be in (0, 60]")
            new = min(max_new, max(8, round(sec * fps / 8) * 8))
            if (src_frames + new - 1) / fps > MAX_TOTAL_S:
                raise BadRequest("the extended video would be longer than %g s" % MAX_TOTAL_S)
            frames = overlap + new
            ext = {"extend_from": src_raw, "src_path": src["path"], "src_frames": src_frames, "overlap": overlap,
                   "total_frames": src_frames + new}
        else:
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
    key = hashlib.sha256(json.dumps([prompt, neg, seed if extra.get("seed") is not None else None, fps, w, h, frames, model,
                                     ext and ext["extend_from"]]).encode()
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
    if ext:
        job.update(ext)
        job["seconds"] = round((ext["total_frames"] - 1) / fps, 2)
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


async def run_ffmpeg(*args, timeout: float = 900) -> str:
    """Run ffmpeg/ffprobe without blocking the event loop; return stdout, raise with the stderr tail on failure."""
    proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("%s timed out" % args[0])
    if proc.returncode != 0:
        print("[ltx-adapter] %s failed: %s" % (args[0], err.decode("utf-8", "replace")[-1500:]), flush=True)
        raise RuntimeError("%s failed" % args[0])
    return out.decode()


async def probe_video(path: str) -> tuple[int, int, float, int, float]:
    """(width, height, fps, frames, duration) from the container metadata, without decoding."""
    info = json.loads(await run_ffmpeg("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", path, timeout=60))
    st = info["streams"][0]
    num, den = (st.get("r_frame_rate") or "24/1").split("/")
    fps = float(num) / float(den or 1)
    dur = float(st.get("duration") or 0)
    frames = int(st["nb_frames"]) if str(st.get("nb_frames", "")).isdigit() else round(dur * fps)
    return int(st["width"]), int(st["height"]), fps, frames, dur


async def source_meta(src: dict) -> tuple[int, int, int, int]:
    """(w, h, fps, frames) of a finished job's video; probed for jobs recovered from disk."""
    if src.get("w") and src.get("h") and src.get("fps"):
        return src["w"], src["h"], src["fps"], src.get("total_frames") or src["frames"]
    w, h, fps, frames, _ = await probe_video(src["path"])
    return w, h, int(round(fps)), frames


async def prepare_overlap(job: dict):
    """Cut the source's last `overlap` frames (and their sound) into the input folder for the ext graph.
    Frames are retimed by number (settb=1/fps, setpts=N) at a constant rate and counted, not cut by -t: a half-frame seek
    margin otherwise shifts the first frame into the next output slot and -t drops the last (seen live, 120 of 121)."""
    fps, ov = job["fps"], job["overlap"]
    start = job["src_frames"] - ov
    ss = max(0.0, (start - 0.5) / fps)
    name = job["raw"] + "_overlap.mp4"
    IN_DIR.mkdir(parents=True, exist_ok=True)
    await run_ffmpeg("ffmpeg", "-v", "error", "-y", "-ss", "%.6f" % ss, "-i", job["src_path"],
                     "-vf", "settb=1/%d,setpts=N" % fps, "-fps_mode", "cfr", "-r", str(fps), "-frames:v", str(ov),
                     "-af", "atrim=start=%.6f:duration=%.6f,asetpts=PTS-STARTPTS" % (start / fps - ss, ov / fps),
                     "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", str(IN_DIR / name))
    got = (await probe_video(str(IN_DIR / name)))[3]
    if got != ov:
        raise RuntimeError("generation failed (could cut only %d of %d overlap frames)" % (got, ov))
    job["overlap_file"] = name


# x264 settings shared by every piece the adapter encodes, so pieces join by stream copy (stitchable=1 keeps the
# headers content-independent); the time base is the frame number at a constant rate.
X264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-x264-params", "stitchable=1", "-pix_fmt", "yuv420p"]


def audio_sidecar(raw: str) -> Path:
    return OUT_DIR / "ltx" / (raw + "_audio.flac")


async def stitch(job: dict):
    """Append the new part of the rendered segment (after the overlap) to the source without re-encoding the source:
    only the new frames are encoded, and the video joins by stream copy (each hop used to re-encode everything, and the
    opening of a long chain lost about 1 dB per hop). Sound lives losslessly in a FLAC sidecar next to each output; the
    seam is crossfaded over XFADE_S with exact lengths (no drift against the picture) and AAC is made once from it.
    The first hop normalizes a ComfyUI-made source once, since its x264 headers differ from ours."""
    fps, ov, seg, raw = job["fps"], job["overlap"], job["path"], job["raw"]
    ltx_dir = OUT_DIR / "ltx"
    src_dur, total_dur = job["src_frames"] / fps, job["total_frames"] / fps
    srcv, newv, lst = (ltx_dir / (raw + suffix) for suffix in ("_srcv.mp4", "_newv.mp4", "_concat.txt"))
    flac, out = audio_sidecar(raw), str(ltx_dir / (raw + "_ext.mp4"))
    vfmt = ["-fps_mode", "cfr", "-r", str(fps), "-an"]
    try:
        if job["src_path"].endswith("_ext.mp4"):  # our own output: copy its video as is
            await run_ffmpeg("ffmpeg", "-v", "error", "-y", "-i", job["src_path"], "-map", "0:v", "-c", "copy", str(srcv))
        else:
            await run_ffmpeg("ffmpeg", "-v", "error", "-y", "-i", job["src_path"], "-vf", "settb=1/%d,setpts=N" % fps,
                             *X264, *vfmt, str(srcv), timeout=1800)
        await run_ffmpeg("ffmpeg", "-v", "error", "-y", "-i", seg, "-vf", "trim=start_frame=%d,settb=1/%d,setpts=N" % (ov, fps),
                         *X264, *vfmt, str(newv))
        lst.write_text("file '%s'\nduration %.6f\nfile '%s'\n" % (srcv, src_dur, newv))
        src_audio = audio_sidecar(Path(job["src_path"]).name.split("_")[0])
        src_in = str(src_audio) if src_audio.exists() else job["src_path"]
        await run_ffmpeg("ffmpeg", "-v", "error", "-y", "-i", src_in, "-i", seg, "-filter_complex",
                         "[0:a]apad,atrim=end={sd:.6f}[a0];[1:a]apad,atrim=start={s1:.6f}:end={e1:.6f},asetpts=PTS-STARTPTS[a1];"
                         "[a0][a1]acrossfade=d={xf:.3f},aresample=48000[a]".format(
                             sd=src_dur, s1=ov / fps - XFADE_S, e1=job["frames"] / fps, xf=XFADE_S),
                         "-map", "[a]", "-c:a", "flac", "-t", "%.6f" % total_dur, str(flac), timeout=1800)
        await run_ffmpeg("ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(flac),
                         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                         "-t", "%.6f" % total_dur, "-movflags", "+faststart", out, timeout=1800)
    finally:
        for p in (srcv, newv, lst):
            p.unlink(missing_ok=True)
    _, _, _, got, dur = await probe_video(out)
    if got != job["total_frames"] or abs(dur - total_dur) > 0.5 / fps:
        raise RuntimeError("generation failed (stitched %d frames / %.3f s, expected %d / %.3f s)"
                           % (got, dur, job["total_frames"], total_dur))
    for p in ltx_dir.glob(raw + "_*_.mp4"):  # ComfyUI's segment(s), including one left by a restart mid-stitch
        p.unlink(missing_ok=True)
    if job.get("overlap_file"):
        (IN_DIR / job["overlap_file"]).unlink(missing_ok=True)
    job["path"] = out


def build_graph(job: dict) -> dict:
    if job.get("extend_from"):
        g = copy.deepcopy(EXT)
        g["500"]["inputs"]["file"] = job["overlap_file"]
    else:
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


async def comfy_queue_state(s: ClientSession, pid: str):
    """"running", "pending" or None (not in ComfyUI's queue)."""
    _, q = await comfy_json(s, "GET", "/queue")
    for state, key in (("running", "queue_running"), ("pending", "queue_pending")):
        if any(len(it) > 1 and it[1] == pid for it in q.get(key, [])):
            return state
    return None


async def comfy_queue_has(s: ClientSession, pid: str) -> bool:
    return await comfy_queue_state(s, pid) is not None


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
                # type "output" only: LoadVideo (extend jobs) reports the clip it loaded as a "type": "input" preview
                if isinstance(f, dict) and f.get("type") == "output" and str(f.get("filename", "")).endswith(".mp4"):
                    return str(OUT_DIR / f.get("subfolder", "") / f["filename"])
    return None


async def run_job(s: ClientSession, job: dict):
    # The job stays "queued" while ComfyUI works on something else (another API job, or a graph from the editor);
    # it turns "in_progress" when ComfyUI starts running it, and JOB_TIMEOUT counts from then.
    job["started"], job["progress"] = time.time(), 0
    # fixed per job and attempt: never submitted twice to the same ComfyUI, and a retry never finds the failed attempt
    pid = str(uuid.UUID(job["raw"][3:]) if not job.get("attempts") else uuid.uuid5(uuid.UUID(job["raw"][3:]), str(job["attempts"])))
    _, h = await comfy_json(s, "GET", "/history/" + pid)
    if pid not in h and not await comfy_queue_has(s, pid):
        st, body = await comfy_json(s, "POST", "/prompt", {"prompt": build_graph(job), "client_id": job["raw"], "prompt_id": pid})
        if st != 200 or body.get("prompt_id") != pid:
            print("[ltx-adapter] %s ComfyUI rejected the graph: %s" % (job["raw"], json.dumps(body)[:2000]), flush=True)
            raise RuntimeError("generation failed (graph rejected)")
    expected = 25 + 0.33 * job["frames"] * job["w"] * job["h"] / (1280 * 704)  # rough seconds, for progress only
    running_since, polls, last_ok = None, 0, time.time()
    while pid not in h:
        await asyncio.sleep(2)
        polls += 1
        now = time.time()
        try:
            _, h = await comfy_json(s, "GET", "/history/" + pid)
            if pid in h:
                break
            if running_since is None or polls % 15 == 0:
                state = await comfy_queue_state(s, pid)
                if state is None:
                    _, h = await comfy_json(s, "GET", "/history/" + pid)  # it may have finished in between
                    if pid in h:
                        break
                    raise RuntimeError("generation failed (ComfyUI lost the job)")
                if state == "running" and running_since is None:
                    running_since = now
                    job["status"] = "in_progress"
                    save_jobs()
            last_ok = now
        except (ClientError, asyncio.TimeoutError, ValueError) as e:
            print("[ltx-adapter] %s poll error %r (retrying)" % (job["raw"], e), flush=True)
            if time.time() - last_ok > 120:  # the same 120 s as wait_comfy: restart (and retry the job once)
                job["wedged"] = "ComfyUI stopped answering"
                raise RuntimeError("generation failed (ComfyUI stopped answering)")
            continue
        if running_since is None:
            if now - job["started"] > QUEUE_WAIT:  # ComfyUI is busy, not stuck: give up on this job only
                await comfy_cancel(s, pid)
                raise RuntimeError("generation failed (ComfyUI was busy with other work for %d s)" % QUEUE_WAIT)
            continue
        job["progress"] = min(95, int(100 * (now - running_since) / max(expected, 1)))
        if now - running_since > JOB_TIMEOUT:  # ~6x the slowest normal render: something is stuck, so restart after failing it
            released = await comfy_cancel(s, pid)
            job["wedged"] = "a job timed out" + ("" if released else " and ComfyUI did not release it")
            raise RuntimeError("generation timed out after %d s" % JOB_TIMEOUT)
    entry = h[pid]
    st = entry.get("status", {})
    if st.get("status_str") == "error":
        err = next((m[1] for m in st.get("messages", []) if m[0] == "execution_error"), {})
        print("[ltx-adapter] %s ComfyUI error: %s" % (job["raw"], json.dumps(err)[:2000]), flush=True)
        if STICKY_CUDA.search(str(err.get("exception_message", ""))):
            job["wedged"] = "a sticky CUDA error"
        raise RuntimeError("generation failed (%s)" % (err.get("node_type") or "execution error"))
    path = harvest(entry)
    if not path:
        raise RuntimeError("generation failed (no video output)")
    job["path"] = path
    try:
        await comfy_json(s, "POST", "/history", {"delete": [pid]})
    except Exception:  # noqa: BLE001
        pass


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
                if job.get("extend_from"):
                    if not os.path.exists(job["src_path"]):
                        raise RuntimeError("generation failed (the source video expired before this job ran)")
                    await prepare_overlap(job)
                await run_job(s, job)
                if job.get("extend_from"):
                    job["progress"] = 97
                    await stitch(job)
                job["status"], job["progress"] = "completed", 100
            except Exception as e:  # noqa: BLE001
                job["status"], job["error"] = "failed", str(e) or type(e).__name__
                if job.get("extend_from"):  # its overlap cut, segment, sidecar or a bad stitch are of no use
                    if job.get("overlap_file"):
                        (IN_DIR / job["overlap_file"]).unlink(missing_ok=True)
                    for p in list((OUT_DIR / "ltx").glob(raw + "_*.mp4")) + [audio_sidecar(raw)]:
                        p.unlink(missing_ok=True)
                    job.pop("path", None)
            why = job.pop("wedged", None)
            if why in RETRY_REASONS and job.get("attempts", 0) < 1:
                # The GPU, not the request, failed (e.g. a Windows TDR reset the eGPU): run the job again after the restart.
                job["attempts"] = job.get("attempts", 0) + 1
                job["status"], job["progress"] = "queued", 0
                job.pop("error", None)
            else:
                job["completed_at"] = int(time.time())
            save_jobs()
            print("[ltx-adapter] %s %s %dx%d %df in %.0f s %s" % (raw, job["status"], job["w"], job["h"], job["frames"],
                  max(0, time.time() - job.get("started", time.time())), job.get("error") or ""), flush=True)
            if why:
                print("[ltx-adapter] %s; exiting so the container restarts%s" % (
                    why, " (the job runs again)" if job["status"] == "queued" else ""), flush=True)
                os._exit(1)


def expire(job: dict):
    for p in (job.get("path"), str(IN_DIR / job["image"]) if job.get("image") else None,
              str(audio_sidecar(job["raw"])) if job.get("raw") else None):
        if p:
            Path(p).unlink(missing_ok=True)
    job["status"] = "expired"


def janitor_pass():
    now = time.time()
    changed = False
    # sources that queued or running extend jobs will still read (their overlap, then the stitch)
    needed = {j["src_path"] for j in jobs.values() if j.get("extend_from") and j["status"] in ("queued", "in_progress")}
    for raw, job in list(jobs.items()):
        done = job.get("completed_at")
        if not done:
            continue
        if job["status"] == "completed" and now - done > KEEP_HOURS * 3600 and job.get("path") not in needed:
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
    sidecars = 0
    for p in list((OUT_DIR / "ltx").glob("*.mp4")) + list((OUT_DIR / "ltx").glob("*_audio.flac")):
        try:
            st = p.stat()
        except OSError:
            continue
        if p.suffix == ".flac":  # counted, but evicted only together with its video (expire)
            sidecars += st.st_size
        else:
            files.append((st.st_mtime, st.st_size, p))
    files.sort()
    total = sum(f[1] for f in files) + sidecars
    for mtime, size, p in files:
        if total <= KEEP_BYTES:
            break
        if now - mtime < 3600 or str(p) in needed:
            continue
        j = by_path.get(str(p))
        if j:
            sc = audio_sidecar(j["raw"])
            size += sc.stat().st_size if sc.exists() else 0
            expire(j)
            changed = True
        else:
            p.unlink(missing_ok=True)
        total -= size
    for p in (list(IN_DIR.glob("ltx*.png")) + list(IN_DIR.glob("ltx*_overlap.mp4")) + list((OUT_DIR / "ltx").glob("ltx*.mp4"))
              + list((OUT_DIR / "ltx").glob("ltx*_audio.flac"))):
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
