#!/usr/bin/env bash
# Compare quantization arms (nvfp4 / fp8 / bf16) on an AGENTIC task, measuring
# "wandering" rather than single-turn code quality.
#
# The question this answers: given the same task from the same starting state,
# does precision change how many moves the agent needs to reach a verified
# success? Tool-call count to a machine-checked pass is the metric; code
# aesthetics are not.
#
# WHY IT IS BUILT THIS WAY -- three traps this design exists to avoid, all of
# which were hit for real on 2026-08-30:
#
#   1. PREFIX-CACHE / STATE CONTAMINATION. Replaying a task set in a fixed order
#      lets run N benefit from runs 1..N-1 (warm prefix cache, artifacts already
#      installed in ~/.m2). An MTP sweep earlier that day produced a monotonic
#      "improvement" that was pure cache warmth: 390 -> 487 -> 516 -> 594 tok/s
#      on the SAME arm. Hence reset_state() before every single run.
#   2. ARM-CORRELATED DRIFT. Running all of arm A then all of arm B confounds
#      precision with anything that changes over hours. Hence block interleaving.
#   3. SILENT PARTIAL FAILURE. A run that dies early looks like a fast run.
#      Hence an explicit machine-checked success gate, and non-converged runs
#      are reported separately rather than averaged in.
#
# Metrics are NOT parsed from harness output -- they are read from Langfuse,
# which already records tokens, cost and tool calls per request. Each arm serves
# under its own model name, so traces self-label and no timestamp slicing is
# needed (timestamp slicing produced a wrong attribution once already).
#
# Usage:
#   quant-arms.sh --dry-run                     show the plan, touch nothing
#   quant-arms.sh --arms fp8,nvfp4 --runs 20    20 runs per arm, blocks of 5
#   quant-arms.sh --task edo-alameda-input-adapter
set -uo pipefail

# --- knobs -------------------------------------------------------------------
ARMS="fp8,nvfp4"
RUNS=20                 # per arm; 25-30 needed to see a 20% effect (see NOTES)
BLOCK=5                 # interleave granularity
TASK_REPO="${TASK_REPO:-edo-alameda-input-adapter}"
PROJECT="$HOME/projects/java-faker-via-deepseek-harness"
REMOTE="${REMOTE:-vmihaylov@mikhailov.tech}"
SSH_PORT="${SSH_PORT:-2222}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/mikhailov_key}"
OUT="${OUT:-$HOME/quant-arms-$(date +%Y%m%d-%H%M%S)}"
DRY=0

# The task. ONE fixed string, byte-identical across every run and every arm.
# Any variation here invalidates the comparison.
read -r -d '' PROMPT <<'TASK' || true
Make the repository repos/REPO_PLACEHOLDER compile, including its test sources.
Missing ru.nsd.* artifacts must be satisfied by building local surrogate jars
into ~/.m2. Do not modify any file under repos/. When you believe it compiles,
verify it yourself and report the exact command you used.
TASK

while [ $# -gt 0 ]; do
  case "$1" in
    --arms)    ARMS="$2"; shift 2 ;;
    --runs)    RUNS="$2"; shift 2 ;;
    --block)   BLOCK="$2"; shift 2 ;;
    --task)    TASK_REPO="$2"; shift 2 ;;
    --out)     OUT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
PROMPT="${PROMPT//REPO_PLACEHOLDER/$TASK_REPO}"

rsh() { ssh -i "$SSH_KEY" -p "$SSH_PORT" -o ConnectTimeout=20 "$REMOTE" "$@"; }
say() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/log"; }

# --- arm switching -----------------------------------------------------------
# Each arm = a served model name + an apply script on the engine host. LiteLLM's
# model_list must name the same model or every request 400s, so it is rewritten
# and restarted in the same step.
arm_model() { case "$1" in
  nvfp4) echo qwen-3.8-27b-nvfp4 ;; fp8) echo qwen-3.8-27b-fp8 ;;
  bf16)  echo qwen-3.8-27b-bf16  ;; *) echo "unknown arm: $1" >&2; return 1 ;;
esac; }

arm_script() { case "$1" in
  nvfp4) echo '~/litellm/rollback/restore-nvfp4.sh' ;;
  fp8)   echo '~/litellm/rollback/apply-fp8.sh'     ;;
  bf16)  echo '~/litellm/rollback/apply-bf16.sh'    ;;
esac; }

switch_arm() {
  local arm="$1" model script
  model="$(arm_model "$arm")" || return 1
  script="$(arm_script "$arm")"
  say "switching engine -> $arm ($model)"
  [ "$DRY" = 1 ] && return 0

  rsh "$script >/dev/null 2>&1" || { say "  ENGINE START FAILED"; return 1; }

  # LiteLLM routes on the body's model name; point it at this arm.
  rsh "cd ~/litellm && sed -i -E 's|qwen-3\.8-27b-(nvfp4|fp8|bf16)|$model|g' config.yaml && docker restart litellm >/dev/null"

  # meterproxy routes on the URL prefix; keep it aligned so the public path works.
  rsh "CK=\$(docker inspect meterproxy --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^CLIENT_KEYS=' | cut -d= -f2-);
       docker rm -f meterproxy >/dev/null 2>&1;
       docker run -d --name meterproxy --network proxy-net --restart unless-stopped \
         -p 127.0.0.1:8091:8091 -e PORT=8091 -e WINDOW_S=1800 \
         -e ROUTES='{\"/$model\": \"litellm:4000\", \"/paddleocr-vl-1.6\": \"litellm:4000\"}' \
         -e UPSTREAM_KEY='' -e CLIENT_KEYS=\"\$CK\" meterproxy-rs:local >/dev/null"

  # the harness talks to the public path, so it must follow the rename too
  python3 - "$model" <<'PY'
import re, sys, os
p = os.path.expanduser("~/.dsh/settings.yaml")
s = open(p).read()
s = re.sub(r'qwen-3\.8-27b-(nvfp4|fp8|bf16)', sys.argv[1], s)
open(p, "w").write(s)
PY

  # health gate: an arm that is not actually serving must not silently produce
  # "fast" runs full of connection errors.
  local i
  for i in $(seq 1 60); do
    if rsh "docker run --rm --network proxy-net curlimages/curl:latest -s -o /dev/null -m 5 -w '%{http_code}' http://litellm:4000/health/liveliness" 2>/dev/null | grep -q 200; then
      say "  arm $arm healthy after ~$((i*5))s"; return 0
    fi
    sleep 5
  done
  say "  ARM $arm NEVER BECAME HEALTHY"; return 1
}

# --- per-run state reset -----------------------------------------------------
# Without this, run N inherits run N-1's installed artifacts and warm prefix
# cache, and the arm that happens to go second looks better.
reset_state() {
  [ "$DRY" = 1 ] && return 0
  rm -rf "$PROJECT/fakes" "$PROJECT/tools" "$PROJECT/gradle-init" 2>/dev/null
  git -C "$PROJECT/repos/$TASK_REPO" reset --hard -q 2>/dev/null
  git -C "$PROJECT/repos/$TASK_REPO" clean -xdfq 2>/dev/null
  rm -rf "$HOME/.m2/repository/ru/nsd" 2>/dev/null
  rsh "docker exec inference-vllm curl -s -X POST http://localhost:8000/reset_prefix_cache >/dev/null 2>&1" || true
}

# --- success gate ------------------------------------------------------------
# Binary and machine-checked. Not "did it say it worked".
verify() {
  local d="$PROJECT/repos/$TASK_REPO"
  [ -d "$d" ] || { echo 0; return; }
  if [ -f "$d/build.gradle" ]; then
    ( cd "$d" && ./gradlew --offline -q compileJava compileTestJava >/dev/null 2>&1 ) && echo 1 || echo 0
  else
    ( cd "$d" && mvn -B -o -q -DskipTests test-compile >/dev/null 2>&1 ) && echo 1 || echo 0
  fi
}

# --- metrics -----------------------------------------------------------------
# Read from Langfuse, sliced by MODEL NAME (self-labelling), never by timestamp.
metrics_for() {
  local model="$1" t0="$2" t1="$3"
  rsh "CU=\$(grep '^CLICKHOUSE_USER=' ~/langfuse/.env | cut -d= -f2);
       CP=\$(grep '^CLICKHOUSE_PASSWORD=' ~/langfuse/.env | cut -d= -f2);
       cat > /tmp/m.sql <<SQL
SELECT count(), sum(length(tool_call_names)),
       toUInt64(sum(usage_details['input'])), toUInt64(sum(usage_details['output'])),
       toUInt64(sum(usage_details['output_reasoning_tokens'])),
       round(sum(cost_details['total']),4)
FROM events_core
WHERE parent_span_id='' AND provided_model_name='$model'
  AND start_time BETWEEN toDateTime($t0) AND toDateTime($t1)
FORMAT TSV
SQL
       docker cp /tmp/m.sql langfuse-clickhouse-1:/tmp/m.sql >/dev/null 2>&1;
       docker exec -e CU=\"\$CU\" -e CP=\"\$CP\" langfuse-clickhouse-1 \
         sh -c 'clickhouse-client --user \"\$CU\" --password \"\$CP\" --queries-file /tmp/m.sql' 2>/dev/null"
}

# --- main --------------------------------------------------------------------
mkdir -p "$OUT"
IFS=',' read -ra ARM_LIST <<< "$ARMS"
CSV="$OUT/results.csv"
echo "arm,run,ok,wall_s,llm_calls,tool_calls,input_tok,output_tok,reasoning_tok,usd" > "$CSV"

say "arms=$ARMS runs=$RUNS block=$BLOCK task=$TASK_REPO out=$OUT"
say "task prompt (fixed for every run):"; printf '%s\n' "$PROMPT" | sed 's/^/    /' | tee -a "$OUT/log"
if [ "$DRY" = 1 ]; then
  say "DRY RUN — plan only:"
  for ((b=0; b*BLOCK < RUNS; b++)); do
    for arm in "${ARM_LIST[@]}"; do
      say "  block $((b+1)): switch to $arm, then runs $((b*BLOCK+1))..$(( (b+1)*BLOCK < RUNS ? (b+1)*BLOCK : RUNS ))"
    done
  done
  say "total engine swaps: $(( (RUNS + BLOCK - 1) / BLOCK * ${#ARM_LIST[@]} ))"
  exit 0
fi

for ((b=0; b*BLOCK < RUNS; b++)); do
  for arm in "${ARM_LIST[@]}"; do
    switch_arm "$arm" || { say "skipping block for $arm"; continue; }
    model="$(arm_model "$arm")"
    hi=$(( (b+1)*BLOCK < RUNS ? (b+1)*BLOCK : RUNS ))
    for ((r=b*BLOCK+1; r<=hi; r++)); do
      reset_state
      t0=$(date +%s); start=$(date +%s)
      ( cd "$PROJECT" && npx --yes @deepseek-ai/dsh --profile headless "$PROMPT" ) \
        > "$OUT/${arm}-run${r}.out" 2>&1
      t1=$(date +%s); wall=$((t1-start))
      ok=$(verify)
      sleep 12   # let the OTel batch exporter flush before reading metrics
      read -r calls tools intok outtok reastok usd < <(metrics_for "$model" "$t0" "$((t1+30))")
      echo "$arm,$r,$ok,$wall,${calls:-0},${tools:-0},${intok:-0},${outtok:-0},${reastok:-0},${usd:-0}" >> "$CSV"
      say "  $arm run $r: ok=$ok wall=${wall}s tools=${tools:-?} \$${usd:-?}"
    done
  done
done

say "done -> $CSV"
python3 - "$CSV" <<'PY'
import csv, statistics as st, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.defaultdict(list)
for r in rows: by[r["arm"]].append(r)
print(f"\n  {'arm':8}{'n':>4}{'pass':>7}{'tool_calls (median)':>22}{'wall_s':>10}{'usd':>9}")
for arm, rs in by.items():
    ok = [r for r in rs if r["ok"] == "1"]
    tc = [int(r["tool_calls"]) for r in ok] or [0]
    wl = [int(r["wall_s"]) for r in ok] or [0]
    ud = [float(r["usd"]) for r in ok] or [0]
    print(f"  {arm:8}{len(rs):>4}{len(ok)}/{len(rs):>5}{st.median(tc):>22.0f}{st.median(wl):>10.0f}{st.mean(ud):>9.3f}")
print("\n  NOTE: medians over PASSING runs only; failures counted separately.")
print("  With n<25/arm only a large (>20%) difference is detectable. A null")
print("  result here means 'not detectable at this n', not 'no difference'.")
PY
