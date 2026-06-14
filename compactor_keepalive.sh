#!/usr/bin/env bash
# frog's-eye: keep the observability condenser alive (auto-restart on exit).
COMP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/compactor_v3.py"
LOG=/var/log/observe/compactor.log
while true; do
  echo "[keepalive] starting compactor at $(date -u +%FT%TZ)" >> "$LOG"
  python3 "$COMP" >> "$LOG" 2>&1
  echo "[keepalive] compactor exited rc=$? at $(date -u +%FT%TZ), restart in 5s" >> "$LOG"
  sleep 5
done
