# dsh-skill — teaching agents to make LTX-2.5 videos

This directory teaches coding agents to make videos with LTX-2.5 (see `../README.md`). A request like "make me a short
video of a cat chasing a butterfly" then ends with an mp4 in the agent's working directory.

| path | what it is |
|---|---|
| `ltx-video/SKILL.md` | The skill: when to use it, how to write an LTX prompt (one paragraph covering subject, setting, camera, look and **sound**), limits, timing, and delivery. |
| `ltx-video/scripts/ltx_video.py` | Standard-library CLI: create, wait, download, with `--resume`. Used by agents without the tool (Kimi, terminals). |
| `plugin/ltx-video/index.js` | A dsh plugin that registers the `ltx_video` tool. dsh uses this instead of the script. |

## Why dsh gets a tool and not just the script

dsh strips key-like variables from the model's shell and runs it sandboxed. In a real dsh-web run, the agent following
the script-only skill had no gateway key. It then spent ten minutes probing `/proc` and the filesystem for one.

The plugin solves this by running inside the dsh process. It resolves the key there through `ctx.credentials`, the
same reference the LLM provider uses, so the model and its shell never see a key. It uses only Node built-ins, so one
file runs in dsh-web (dsh 0.2.0-rc.2) and in the macOS app (0.2.2-rc.1).

## Install

| where | skill | tool |
|---|---|---|
| dsh-web on mh (container `dsh-web`) | `/root/.dsh/skills/ltx-video/` | `/root/.dsh/plugins/ltx-video/` plus a row in `profiles/web/cordis.patch.yml`, `keyRef: MIKHAILOV_JAVAFAKER_KEY`. The profile reloads live. |
| DeepSeek Harness.app (Mac) | `~/.dsh/skills/ltx-video/` | `~/.dsh/plugins/ltx-video/` plus a row in `profiles/desktop/cordis.patch.yml`, `keyRef: MIKHAILOV_API_KEY`. Restart the app to load it. |
| Kimi Code on the Strix Halo | `%USERPROFILE%\.kimi-code\skills\ltx-video\` | none; Kimi runs the script with `py -3` |

The profile row (an absolute `name`):

```yaml
- insert:
    - id: ltx-video
      name: /root/.dsh/plugins/ltx-video/index.js
      config:
        keyRef: MIKHAILOV_JAVAFAKER_KEY   # optional: baseUrl, maxWaitSeconds (900)
```

dsh watches its skill roots, so a new or edited `SKILL.md` reaches the next model step without a restart.

## Where the script sends requests

The script picks its endpoint from what it finds, in this order:
1. **`LTX_API_KEY` is set:** the gateway, `https://inference.mikhailov.tech/ltx-2.5/v1`, with that key. `LTX_BASE_URL`
   overrides the URL.
2. **`~/.ltx/engine.env` exists** (the GPU box itself): the local adapter, `127.0.0.1:8101`, with that box's engine
   key. Going out to the gateway and back from the Halo loops through the home router: a 0.6 MB clip took 59 s that
   way, against 25 s end to end locally.
3. **Otherwise:** the gateway, with `$MIKHAILOV_API_KEY`, `$MIKHAILOV_JAVAFAKER_KEY`, or the `refs` in
   `~/.dsh/.credentials.yaml`.

On Windows, call it with `py -3`. A plain `py` follows the script's `#!/usr/bin/env python3` line to the Microsoft Store
alias.

## Verified (2026-11-22)

- **dsh-web, headless run of the live profile with the plugin:**
  - The request was "Please make me a short video (about 3 seconds) of a cat chasing a butterfly in a sunny garden,
    generated with LTX."
  - The agent loaded the skill, wrote the prompt, called `ltx_video`, and saved a 3 s 1280×704 clip.
  - It reported the seed and the AI-disclosure note.
- **Plugin under a stub dsh context:**
  - text-to-video and image-to-video, where a portrait image gave a portrait size;
  - a 400 passed through with the adapter's own message;
  - a missing key reference and a missing prompt each give a clear error.
- **Script:**
  - Mac to the gateway: text-to-video, image-to-video, `--resume` after `--max-wait`, unknown id, and 400;
  - Halo to the local adapter, with `py -3`.
- **Kimi Code 0.41 on the Halo** discovered and loaded the skill. The run then died on Kimi's own OAuth refresh:
  "self-signed certificate in certificate chain", most likely from TLS interception on that box. That is not a skill
  problem.
