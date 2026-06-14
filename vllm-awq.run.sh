#!/usr/bin/env bash
# frog's-eye compacting model: Qwen 3.6 35B-A3B AWQ (4-bit MoE, ~3B active) on GPU1, serving the
# condenser. Reachable via the shared gateway route /qwen-3.6-35b-a3b-awq/v1 (Caddy -> obs-vllm:8000).
set -e
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; . "$D/.env"; set +a
KEY="${OBSERVABILITY_COMPACTOR_API_KEY:-$PROPOSER_API_KEY}"
docker rm -f obs-vllm omega-reptile-vllm-playground >/dev/null 2>&1 || true
docker run -d --name obs-vllm --restart unless-stopped \
  --runtime nvidia --ipc host --network proxy-net \
  -e NVIDIA_VISIBLE_DEVICES=GPU-aa391cd3-2b77-0ded-a4cf-b2ee93f9b846 \
  -v /home/vmihaylov/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.20.0-cu130-ubuntu2404 \
  --model cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit --served-model-name qwen-3.6-35b-a3b-awq \
    --max-model-len 65536 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --max-num-seqs 128 \
    --enforce-eager --enable-prefix-caching --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.95 --host 0.0.0.0 --port 8000 --api-key="$KEY"
echo "obs-vllm (re)created"
