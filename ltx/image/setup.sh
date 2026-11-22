#!/bin/bash
# Builds the ltx-comfy environment inside a python:3.12 container (no Docker Hub pulls; committed afterwards).
set -euo pipefail
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore DEBIAN_FRONTEND=noninteractive
step() { echo "STEP $(date -u +%H:%M:%S) $*"; }
step apt; apt-get update -qq && apt-get install -y -qq --no-install-recommends git ffmpeg >/dev/null && rm -rf /var/lib/apt/lists/*
step torch; pip install -q torch==2.14.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__)" > /tmp/torch_before; step "torch $(cat /tmp/torch_before)"
step comfyui; git clone -q https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI; cd /opt/ComfyUI; git checkout -q v0.45.0; git rev-parse HEAD > COMFY_COMMIT
grep -vE '^(torch|torchvision|torchaudio)([<>=~! ].*)?$' requirements.txt > /tmp/req.txt; pip install -q -r /tmp/req.txt
step gguf; git clone -q https://github.com/city96/ComfyUI-GGUF custom_nodes/ComfyUI-GGUF; cd custom_nodes/ComfyUI-GGUF; git checkout -q 6ea2651e7d; pip install -q -r requirements.txt; pip install -q aiohttp
python -c "import torch; b=open('/tmp/torch_before').read().strip(); assert torch.__version__==b, (b, torch.__version__); print('torch kept', torch.__version__, 'cuda', torch.version.cuda)"
cp /b/extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml
mkdir -p /opt/service /data/out /data/input /data/user /data/temp; cp /b/entrypoint.sh /opt/service/entrypoint.sh; chmod +x /opt/service/entrypoint.sh
step "SETUP-OK comfy $(cat /opt/ComfyUI/COMFY_COMMIT | cut -c1-12)"
