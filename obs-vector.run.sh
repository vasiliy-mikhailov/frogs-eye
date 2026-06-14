#!/usr/bin/env bash
# frog's-eye capture: (re)create the obs-vector container reproducibly. Generic — watches the
# whole host: /var/log/**/* + /var/lib/docker/containers + host_metrics, sinks to /var/log/observe.
set -e
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker rm -f obs-vector >/dev/null 2>&1 || true
mkdir -p /home/vmihaylov/vector-data
docker run -d --name obs-vector --restart unless-stopped \
  -v "$D/vector.toml":/etc/vector/vector.toml:ro \
  -v /var/log:/var/log:ro \
  -v /var/log/observe:/var/log/observe:rw \
  -v /var/lib/docker/containers:/var/lib/docker/containers:ro \
  -v /home/vmihaylov/vector-data:/var/lib/vector \
  timberio/vector:0.39.0-alpine --config /etc/vector/vector.toml
echo "obs-vector (re)created"
