---
name: ltx-video
description: Generate a video clip with sound (up to 20 s per render, up to 1280x704, extendable to minutes) from a text prompt, or animate a still image, with LTX-2.5 running on the Strix Halo's RTX 4090 behind inference.mikhailov.tech. Use when the user asks to make, render or generate a video, movie, clip, animation or "LTX" output.
---

# LTX-2.5 video generation

LTX-2.5 (Lightricks, 22B, distilled) renders video **with an audio track** on one RTX 4090 in the Strix Halo box.
The script `scripts/ltx_video.py` in this skill's folder calls `https://inference.mikhailov.tech/ltx-2.5/v1` (the
OpenAI videos API). It creates the job, waits for it, and saves the mp4. It finds the gateway key by itself; never print,
ask for or pass a key.

## Run it

**In dsh, call the `ltx_video` tool.** It creates the job, waits (about a minute), and saves the mp4 in the working
directory. It handles the gateway key itself: never look for, print or ask for a key. Arguments are `prompt`, `seconds`,
`size`, `image`, `seed`, `negative_prompt`, `out`, `extend_from` and `resume_id`.

**Without that tool** (Kimi, a terminal), run the script that sits next to this file, `scripts/ltx_video.py`, with
`python3`, or `py` on Windows (the Strix Halo has `py`, not `python`). It finds its key by itself; on the Strix Halo it talks to the local adapter directly:

```bash
python3 <this skill's folder>/scripts/ltx_video.py --prompt "<prompt>" --seconds 5 --out videos/<short-name>.mp4
```

Give the shell call a timeout longer than the script's `--max-wait` (default 900 s), so about 16 minutes. If your shell
caps lower, pass `--max-wait` about 60 s below that cap. The script prints the full video id as soon as the job is
queued. If it exits with code 3, or is cut off, run `--resume <that id>`; do not render again.

| option | meaning |
|---|---|
| seconds | 1–20, default 5 (24 fps); with extend, the new seconds, 1–15, default 15 |
| size | `1280x704` landscape (default), `704x1280` portrait, `896x896` square; larger sizes are scaled down |
| image | image-to-video: a PNG/JPEG/WebP that becomes the first frame; the size then follows its aspect ratio |
| seed | reproduce a result; the tool and the script report the seed they used |
| negative prompt | what to avoid; the default is fine for most clips |
| out | where to save (default `videos/ltx-<time>.mp4` under the working directory); never overwrites |

Typical times on the 4090: about 65 s for 5 s at 1280x704, 95 s for 10 s, 170 s for 20 s, and about 3.5 min per
15 s extension. The GPU renders **one clip at a
time**, and a full queue is waited out automatically. Make several clips one after another, never in parallel. If a
call stops waiting, continue with `resume_id` (tool) or `--resume` (script); do not create the clip again.

## Longer videos (extend)

Over 20 s, make the opening clip, then **extend** it: tool `extend_from: <video id>`, or script `--extend <video id>`.
- The last 5 s of picture and sound carry over, so the continuation is seamless.
- Your prompt says **what happens next**, written in the same style and with the same characters.
- Each call adds up to 15 s, and its result is the **whole** video so far. Extend that result's id again to go on,
  up to 15 minutes.
- Don't download the intermediate hops, since each result already contains the earlier ones. Pass `download: false`
  (tool) or `--no-download` (script) on every hop except the last, and present only the final video.
- A video id can be extended for up to 24 h after it finished, or less when the server's store is full. After that,
  start a new opening clip.
- Size and frame rate follow the source.

Plan long pieces as a sequence of beats: one prompt per 15 s, each continuing the last.
- Restate the look and the key characters in every prompt; the model only sees the last 5 s.
- Expect slow drift over many hops in faces, clothes and sets.
- For a film with cuts, separate clips joined afterwards often work better than one endless extension.

## Write the prompt

Turn the user's idea into **one flowing paragraph in English** (translate it if they wrote in another language),
60–150 words, describing what the camera sees over the whole clip:

1. The subject and what it does, with concrete motion verbs. Keep the action small enough to fit the duration.
2. The setting, time of day and lighting.
3. The camera: static, slow dolly-in, pan, tracking, handheld, aerial, close-up or wide.
4. The look: photoreal or cinematic, film grain, lens, colour palette.
5. **The sound:** ambient noise, music, effects; put any spoken line in quotes, with who says it.

Describe what is there rather than what is not. Avoid lists, camera cuts, on-screen text and more than one or two
characters. For image-to-video, describe the motion and sound that should follow from the image rather than
re-describing it.

Example: *A red paper boat drifts down a rain-filled street gutter at dusk, spinning slowly as it passes glowing shop
windows. Raindrops ripple the water and the reflections of neon signs shimmer around the boat. Low tracking shot at
water level that follows the boat, shallow depth of field, cinematic colour grading. Sound of steady rain, distant
traffic and water gurgling into a drain.*

## Deliver it

After the script prints `saved <path>`, call the `present` tool with that path and a one-line description. Then tell the
user the prompt you used, the length and size, and the seed, so they can ask for a variation. Mention that the clip is
AI-generated: the LTX-2.x Community License requires disclosure wherever it is shared.

## When it fails

| result | what to do |
|---|---|
| HTTP 400 (script: exit 2) | Fix the arguments the message names (size, seconds, image) and retry. |
| `generation failed` | Retry once. A second failure usually means the GPU box is busy or restarting: tell the user, and don't loop. |
| HTTP 5xx, connection errors, `video_not_found` | The Halo may be rebooting. Wait about a minute, retry once, then report. |
| still running, or the download failed | Continue with `resume_id` (tool) or the printed `--resume` command (script). Never render it again. |
| "extend_from must name a completed video…" | The source expired (24 h) or is not finished yet. Wait for it, or start a new opening clip. Don't retry the same id, and don't pass a file path. |
