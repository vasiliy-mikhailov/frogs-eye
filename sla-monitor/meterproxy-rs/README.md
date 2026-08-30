# meterproxy-rs

A Rust port of `../meterproxy/meterproxy.py`, built to be a **drop-in replacement**:
same environment variables, same routes, same dashboard, and — the part that
matters — the same Prometheus series, so the existing Grafana panels do not
notice the swap.

## Why port it

Not for speed. The measurements below say plainly that throughput and latency
are unchanged, because an inference request is dominated by the engine, not by
the proxy in front of it.

The reasons that do hold up:

- **It sits in the hot path of every byte of every stream.** Once LiteLLM is
  behind it, this is the only component left that must parse each SSE chunk of
  every response.
- **No GIL.** On 2026-08-30 this estate lost 3.5 hours to a pure-Python hot loop
  inside vLLM's tool-call parser, which held the GIL and froze every endpoint on
  a single asyncio loop — including `/v1/models`, which touches no model at all.
  The Python meter is the same *shape* of risk. Rust removes that class of
  failure from the meter itself.
- **Footprint.** 53 MB resident becomes 12 MB, on a box that also hosts a
  saturated GPU workload and ~30 other containers.

## Measured, on the real workload

Both versions were run side by side on `proxy-net`, fed identical traffic, both
proxying to the same LiteLLM instance and the same vLLM engine.

| | Python | Rust |
|---|---|---|
| Prometheus series emitted | 67 | **67, names identical** |
| Resident memory | 53.6 MB | **12.3 MB** |
| CPU, 12 × 234 KB requests | 0.030 s | 0.010 s |
| CPU, 8 concurrent × 1500-token streams (~12k tokens) | 0.230 s | **0.150 s** |
| Wall clock, same test | 20.62 s | 20.40 s |

The wall-clock figures are the honest headline: **there is no speed win.** At
production scale the CPU saved is a few minutes over a 35-hour window. Take this
port for the footprint and the removal of a GIL-bound event loop, not for
latency.

Functional parity was verified rather than assumed — streamed tool calls
(5 deltas, `finish_reason: tool_calls`), the `[DONE]` terminator, the 57k-token
prompt path, and both directions of the usage-chunk rule: stripped when the
proxy injected `include_usage`, preserved when the client asked for it itself.

## Configuration

Identical to the Python version.

| Variable | Meaning |
|---|---|
| `PORT` | listen port (default 8091) |
| `ROUTES` | JSON object, url-prefix → `host:port`. Longest prefix wins. |
| `CLIENT_KEYS` | JSON object, consumer key → label. Labels become the `key` metric label. |
| `UPSTREAM_KEY` | if non-empty, replaces the consumer's bearer on the way upstream. Leave **empty** when a gateway behind this proxy needs to see the consumer's own key. |
| `WINDOW_S` | dashboard sliding window, seconds (default 1800) |
| `MAXREC` | in-memory record cap (default 200000) |

Upstreams are addressed as `http://host:port`, as in the original. There is no
TLS client: the public-internet hop belongs to the gateway behind this proxy.

## Build and run

```
docker build -t meterproxy-rs:local .
docker run -d --name meterproxy --network proxy-net \
  -e PORT=8091 \
  -e ROUTES='{"/qwen-3.8-27b-nvfp4": "litellm:4000"}' \
  -e UPSTREAM_KEY='' \
  -e CLIENT_KEYS='{"sk-...":"bump-java"}' \
  meterproxy-rs:local
```

## Deliberate difference from the Python

`# HELP` and `# TYPE` are emitted **once per metric family** rather than once
per series. The original repeated them for every `(key, model)` pair, which is
malformed exposition that Prometheus tolerates. Series identity is unchanged.
