#!/bin/bash
# ComfyUI on all container interfaces (only published to Windows localhost); the video adapter joins later.
set -e
cd /opt/ComfyUI
exec python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch \
  --output-directory /data/out --input-directory /data/input --user-directory /data/user --temp-directory /data/temp \
  --extra-model-paths-config /opt/ComfyUI/extra_model_paths.yaml ${COMFY_EXTRA_ARGS:-}
