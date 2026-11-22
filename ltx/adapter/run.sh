#!/bin/bash
# ltx container entrypoint: ComfyUI + the OpenAI-videos adapter; if either dies, exit so Docker restarts both.
cd /opt/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch \
  --output-directory /data/out --input-directory /data/input --user-directory /data/user --temp-directory /data/temp \
  --extra-model-paths-config /opt/ComfyUI/extra_model_paths.yaml ${COMFY_EXTRA_ARGS:-} &
COMFY=$!
for i in $(seq 1 120); do curl -sf -o /dev/null http://127.0.0.1:8188/system_stats && break; kill -0 $COMFY 2>/dev/null || exit 1; sleep 2; done
python /opt/adapter/ltx_video_adapter.py &
ADAPTER=$!
wait -n $COMFY $ADAPTER
echo "[run.sh] a process exited; stopping the container so Docker restarts it" >&2
kill $COMFY $ADAPTER 2>/dev/null
exit 1
