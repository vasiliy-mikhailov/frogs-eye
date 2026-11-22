// Local dsh plugin: the `ltx_video` tool. It renders a short video with sound through LTX-2.5 on the Strix Halo's
// RTX 4090 (OpenAI videos API behind inference.mikhailov.tech), waits for it and saves the mp4 in the session's
// working directory. The model then delivers it with `present`; the ltx-video skill holds the prompt guidance.
//
// Why a tool and not a shell script: dsh strips key-like variables from the model's shell and sandboxes it, so the
// model never holds a gateway key. This tool resolves the key here, inside the dsh process, through ctx.credentials
// (the same reference the LLM provider uses: an environment variable or a `refs` entry in ~/.dsh/.credentials.yaml),
// and never returns or logs it. Only Node built-ins are used, so the file runs in dsh-web and in the desktop app.
//
// Mount it with a row in the profile's cordis.patch.yml:
//   - insert:
//       - id: ltx-video
//         name: <absolute path of this file>
//         config: { keyRef: MIKHAILOV_API_KEY }     # optional: baseUrl, maxWaitSeconds
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";

export const name = "ltx-video";
export const inject = ["tools"];

const MAX_PIXELS = 1280 * 704;
const UA = "dsh-ltx-video/1.0";

const sleep = (ms, signal) => new Promise((resolve, reject) => {
  if (signal?.aborted) return reject(signal.reason ?? new Error("aborted"));
  const t = setTimeout(resolve, ms);
  signal?.addEventListener("abort", () => { clearTimeout(t); reject(signal.reason ?? new Error("aborted")); }, { once: true });
});

/** (width, height) of a PNG, JPEG or WebP buffer, or undefined. */
function imageSize(b) {
  if (b.length >= 24 && b.readUInt32BE(0) === 0x89504e47) return [b.readUInt32BE(16), b.readUInt32BE(20)];
  if (b[0] === 0xff && b[1] === 0xd8) {
    let i = 2;
    while (i + 9 < b.length) {
      if (b[i] !== 0xff) { i++; continue; }
      const m = b[i + 1];
      if ([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf].includes(m)) return [b.readUInt16BE(i + 7), b.readUInt16BE(i + 5)];
      if (m === 0xd8 || m === 0x01 || (m >= 0xd0 && m <= 0xd7)) { i += 2; continue; }
      i += 2 + b.readUInt16BE(i + 2);
    }
    return undefined;
  }
  if (b.length >= 30 && b.toString("latin1", 0, 4) === "RIFF" && b.toString("latin1", 8, 12) === "WEBP") {
    const kind = b.toString("latin1", 12, 16);
    if (kind === "VP8 ") return [b.readUInt16LE(26) & 0x3fff, b.readUInt16LE(28) & 0x3fff];
    if (kind === "VP8L") { const v = b.readUInt32LE(21); return [(v & 0x3fff) + 1, ((v >>> 14) & 0x3fff) + 1]; }
    if (kind === "VP8X") return [b.readUIntLE(24, 3) + 1, b.readUIntLE(27, 3) + 1];
  }
  return undefined;
}

/** Largest WxH (multiples of 64, within the pixel budget and 1:3) keeping the image's aspect ratio. */
function sizeFor(w0, h0) {
  const ratio = Math.max(1 / 3, Math.min(3, w0 / h0));
  let best = [0, 0, 1280, 704];
  for (let h = 256; h <= 2048; h += 64) {
    const w = Math.round((h * ratio) / 64) * 64;
    if (w < 256 || w * h > MAX_PIXELS || Math.max(w, h) > 3 * Math.min(w, h)) continue;
    const err = Math.abs(w / h - ratio) / ratio;
    if (err <= 0.08 && (w * h > best[0] || (w * h === best[0] && -err > best[1]))) best = [w * h, -err, w, h];
  }
  return `${best[2]}x${best[3]}`;
}

/** The adapter's own message out of LiteLLM's wrapped error text. */
function errorMessage(body) {
  let msg = body?.error?.message ?? body?.error ?? body;
  if (typeof msg !== "string") msg = JSON.stringify(msg);
  const inner = /Exception - (\{.*\})/s.exec(msg);
  if (inner) { try { msg = JSON.parse(inner[1]).error.message; } catch { /* keep the wrapped text */ } }
  return String(msg).slice(0, 400);
}

export function apply(ctx, config) {
  const base = String(config?.baseUrl ?? "https://inference.mikhailov.tech/ltx-2.5/v1").replace(/\/+$/, "");
  const keyRef = String(config?.keyRef ?? "MIKHAILOV_API_KEY");
  const maxWaitMs = Number(config?.maxWaitSeconds ?? 900) * 1000;

  async function gatewayKey() {
    const credentials = ctx.get("credentials");
    const hit = credentials !== undefined ? await credentials.resolve(keyRef) : undefined;
    const value = hit?.value ?? process.env[keyRef];
    if (!value) throw new Error(`ltx_video: the gateway key reference ${keyRef} is not set in this dsh profile`);
    return value;
  }

  async function call(method, route, key, body, signal, raw = false) {
    const res = await fetch(base + route, {
      method, signal,
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", "User-Agent": UA },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const buf = Buffer.from(await res.arrayBuffer());
    if (raw && res.ok) return { status: res.status, data: buf, headers: res.headers };
    let data = {};
    try { data = JSON.parse(buf.toString("utf8")); } catch { data = { error: { message: buf.toString("utf8", 0, 300) } }; }
    return { status: res.status, data, headers: res.headers };
  }

  async function freePath(cwd, wanted) {
    let rel = wanted?.trim() || `videos/ltx-${new Date().toISOString().replace(/[-:]/g, "").replace("T", "-").slice(0, 15)}.mp4`;
    if (!rel.endsWith(".mp4")) rel += ".mp4";
    let target = path.resolve(cwd, rel);
    for (let n = 2; await stat(target).then(() => true, () => false); n++) target = path.resolve(cwd, rel.slice(0, -4) + `-${n}.mp4`);
    await mkdir(path.dirname(target), { recursive: true });
    return target;
  }

  const properties = {
    prompt: { type: "string", description: "One paragraph in English: subject and action, setting and light, camera, look, and the SOUND (ambience, music, quoted speech). Required unless resume_id is given." },
    seconds: { type: "number", description: "Clip length, 1-10 seconds (default 5)." },
    size: { type: "string", description: "WxH: 1280x704 landscape (default), 704x1280 portrait, 896x896 square. With image, the default follows the image's aspect ratio." },
    image: { type: "string", description: "Image-to-video: path (relative to the working directory) of a PNG, JPEG or WebP to use as the first frame." },
    seed: { type: "integer", description: "Fix to reproduce a result; random if omitted (the result reports it)." },
    negative_prompt: { type: "string", description: "What to avoid; rarely needed." },
    out: { type: "string", description: "Where to save the mp4, relative to the working directory (default videos/ltx-<time>.mp4). Never overwrites." },
    resume_id: { type: "string", description: "Wait for and download a job that an earlier call started but could not finish waiting for." },
  };

  ctx.tools.register({
    name: "ltx_video",
    description: "Generate a short video clip WITH SOUND (1-10 s, up to 1280x704) using LTX-2.5 on the Strix Halo's RTX 4090 via inference.mikhailov.tech, wait for it (about 1 minute per 5 s) and save it as an mp4 in the working directory. Text-to-video, or image-to-video from a still. Load the ltx-video skill first for prompt guidance; one render at a time, so make several clips one after another. Deliver the saved file with present.",
    parameters: { type: "object", properties },
    output: {
      schema: {
        type: "object",
        properties: {
          path: { type: "string" }, seconds: { type: "string" }, size: { type: "string" },
          seed: { type: "integer" }, video_id: { type: "string" }, elapsed_s: { type: "number" },
        },
      },
      render: (_args, v) => [{
        type: "text",
        text: `Saved ${v.path} (${v.seconds} s, ${v.size}${v.seed !== undefined ? `, seed ${v.seed}` : ""}, ${v.elapsed_s} s). Deliver it with present. Video id: ${v.video_id}`,
      }],
    },
    async execute(args, exec) {
      const cwd = exec.agent?.session?.header?.cwd;
      if (!cwd) throw new Error("ltx_video needs a session working directory");
      const signal = exec.signal;
      const started = Date.now();
      const key = await gatewayKey();
      let id = typeof args.resume_id === "string" && args.resume_id.trim() ? args.resume_id.trim() : undefined;
      let seed;
      if (!id) {
        if (typeof args.prompt !== "string" || !args.prompt.trim()) throw new Error("ltx_video: prompt is required (or resume_id)");
        seed = Number.isInteger(args.seed) ? args.seed : Math.floor(Math.random() * 2 ** 47) + 1;
        const body = { model: "ltx-2.5", prompt: args.prompt.trim(), seconds: String(args.seconds ?? 5), seed };
        if (typeof args.negative_prompt === "string" && args.negative_prompt.trim()) body.negative_prompt = args.negative_prompt.trim();
        let size = typeof args.size === "string" && args.size.trim() ? args.size.trim() : undefined;
        if (typeof args.image === "string" && args.image.trim()) {
          const img = await readFile(path.resolve(cwd, args.image.trim()));
          if (img.length > 20 * 1024 * 1024) throw new Error("ltx_video: image is larger than 20 MB");
          const dims = imageSize(img);
          const mime = img[0] === 0x89 ? "image/png" : img[0] === 0xff ? "image/jpeg" : "image/webp";
          body.input_reference = `data:${mime};base64,${img.toString("base64")}`;
          if (!size && dims) size = sizeFor(dims[0], dims[1]);
        }
        body.size = size ?? "1280x704";
        for (;;) {
          const r = await call("POST", "/videos", key, body, signal);
          if (r.status === 429 && Date.now() - started < maxWaitMs) {  // one GPU, short queue: wait our turn
            await sleep(Math.min(60, Number(r.headers.get("retry-after")) || 30) * 1000, signal);
            continue;
          }
          if (r.status !== 200 || typeof r.data?.id !== "string") throw new Error(`ltx_video: create failed (HTTP ${r.status}): ${errorMessage(r.data)}`);
          id = r.data.id;
          break;
        }
      }
      let job;
      for (;;) {
        let r;
        try { r = await call("GET", `/videos/${id}`, key, undefined, signal); } catch (e) {
          if (signal?.aborted) throw e;
          r = { status: 0, data: {} };  // a network blip or a restarting box: keep polling
        }
        job = r.data;
        if ((r.status === 200 || r.status === 404) && (job?.status === "completed" || job?.status === "failed")) break;
        if (Date.now() - started > maxWaitMs) throw new Error(`ltx_video: still ${job?.status ?? "unknown"} after ${Math.round((Date.now() - started) / 1000)} s; call ltx_video again with resume_id "${id}"`);
        await sleep(5000, signal);
      }
      if (job.status === "failed") throw new Error(`ltx_video: generation failed: ${job.error?.message ?? "unknown error"} (${job.error?.code ?? "no code"})`);
      const d = await call("GET", `/videos/${id}/content`, key, undefined, signal, true);
      if (d.status !== 200 || !Buffer.isBuffer(d.data) || d.data.toString("latin1", 4, 8) !== "ftyp") throw new Error(`ltx_video: download failed (HTTP ${d.status}); retry with resume_id "${id}"`);
      const target = await freePath(cwd, args.out);
      await writeFile(target, d.data);
      return {
        path: path.relative(cwd, target) || target,
        seconds: String(job.seconds ?? args.seconds ?? ""),
        size: String(job.size ?? ""),
        ...(seed !== undefined ? { seed } : {}),
        video_id: id,
        elapsed_s: Math.round((Date.now() - started) / 1000),
      };
    },
  });
}
