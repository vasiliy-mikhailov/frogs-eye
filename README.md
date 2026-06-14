# frog's-eye — host observability digest

Generic, project-independent. Vector globs `/var/log/**/*` + `/var/lib/docker/containers` +
host/GPU metrics; a compacting LLM condenses the firehose (repetitions collapsed ~100x first) — three components, all here: the AWQ model server (`obs-vllm`), Vector capture (`obs-vector`), and the compactor
into a multi-resolution digest at `~/observe/{fast,medium,slow}.jsonl`. Watches the *whole host*.

## Run
    ./vllm-awq.run.sh             # the AWQ compacting model server (obs-vllm, GPU1)
    ./obs-vector.run.sh           # (re)create the Vector capture container
    setsid nohup ./compactor_keepalive.sh >/dev/null 2>&1 &   # digest compactor (auto-restart)
    setsid nohup ./gpu_metrics.sh >/dev/null 2>&1 &           # GPU sampler -> /var/log/observe/app/gpu.log
    crontab: */20 * * * * /usr/sbin/logrotate -s ~/.logrotate.state <repo>/logrotate.conf

## Config (`.env`, gitignored)
    OBSERVABILITY_COMPACTOR_BASE_URL / _MODEL / _API_KEY   # the compacting LLM endpoint

Any source that wants to be observed just writes under `/var/log` — no per-source wiring.
