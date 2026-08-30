#!/usr/bin/env bash
# Reset the java-faker task to a known clean state and run it once on the
# CURRENTLY SERVED arm, recording milestone timings.
#
# The point is that a run is only comparable to another run if it starts from
# the same place. Today three runs were compared that each began differently:
#   - FP8  started with an EMPTY ~/.m2/repository/ru/nsd and had to synthesise
#     and install all 58 surrogate jars before anything could compile.
#   - BF16 started with those 58 jars already present, so its "first BUILD
#     SUCCESS at +15.9 min" was a head start, not a result.
#   - the repo tree also moved between runs (repos/<name> -> <name>).
# This script removes all three differences.
#
#   remeasure-arm.sh --dry-run     show what would be destroyed, touch nothing
#   remeasure-arm.sh --reset-only  reset state, do not launch the harness
#   remeasure-arm.sh               reset, then run, then report milestones
set -uo pipefail

PROJECT="${PROJECT:-$HOME/projects/java-faker-via-deepseek-harness}"
REMOTE="${REMOTE:-vmihaylov@mikhailov.tech}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/mikhailov_key}"
SSH_PORT="${SSH_PORT:-2222}"
GITLAB="${GITLAB:-https://gitlab.mikhailov.tech}"
DRY=0; RESET_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)    DRY=1; shift ;;
    --reset-only) RESET_ONLY=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

rsh() { ssh -i "$SSH_KEY" -p "$SSH_PORT" -o ConnectTimeout=20 "$REMOTE" "$@"; }
say() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }

# --- which arm is live? the engine is the source of truth --------------------
ARM=$(rsh 'docker inspect inference-vllm --format "{{json .Args}}" | python3 -c "import sys,json;a=json.load(sys.stdin);print(a[a.index(\"--served-model-name\")+1])"' 2>/dev/null)
[ -n "$ARM" ] || { echo "cannot read served model name from the engine" >&2; exit 1; }
say "live arm: $ARM"

# --- what will be destroyed --------------------------------------------------
M2="$HOME/.m2/repository/ru/nsd"
say "reset plan:"
echo "    rm -rf $M2                 ($(find "$M2" -name '*.jar' 2>/dev/null | wc -l | tr -d ' ') surrogate jars)"
echo "    rm -rf $PROJECT/fakes      (generated surrogate sources)"
echo "    rm -rf $PROJECT/tools $PROJECT/gradle-init"
echo "    restore layout: <project>/<repo>  ->  <project>/repos/<repo>"
echo "    git reset --hard + clean -xdf on every repo, on its DEFAULT branch"
echo "    reset the engine prefix cache"
if [ "$DRY" = 1 ]; then say "DRY RUN — nothing touched"; exit 0; fi

# --- reset -------------------------------------------------------------------
rm -rf "$M2" "$PROJECT/fakes" "$PROJECT/tools" "$PROJECT/gradle-init"
mkdir -p "$PROJECT/repos"
# repos drifted to the project root during the last run; put them back
for d in "$PROJECT"/ca2_* "$PROJECT"/edo-*; do
  [ -d "$d" ] || continue
  mv "$d" "$PROJECT/repos/" 2>/dev/null
done
n=0
for d in "$PROJECT"/repos/*/; do
  [ -d "$d/.git" ] || continue
  # DEFAULT branch, never the fakes branch -- using it is what the task forbids
  def=$(git -C "$d" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|origin/||')
  [ -n "$def" ] || def=$(git -C "$d" remote show origin 2>/dev/null | sed -n 's/.*HEAD branch: //p')
  [ -n "$def" ] || def=master
  git -C "$d" checkout -q "$def" 2>/dev/null
  git -C "$d" reset --hard -q 2>/dev/null
  git -C "$d" clean -xdfq 2>/dev/null
  n=$((n+1))
done
say "reset $n repos to their default branch, wiped ~/.m2/ru/nsd and generated dirs"
rsh 'docker exec inference-vllm curl -s -X POST http://localhost:8000/reset_prefix_cache >/dev/null 2>&1' || true

# sanity: the reset must actually have worked, or the run measures nothing
left=$(find "$M2" -name '*.jar' 2>/dev/null | wc -l | tr -d ' ')
[ "$left" = "0" ] || { echo "ABORT: $left surrogate jars survived the reset" >&2; exit 1; }
[ -d "$PROJECT/fakes" ] && { echo "ABORT: fakes/ survived the reset" >&2; exit 1; }
say "verified clean: 0 surrogate jars, no fakes/"
[ "$RESET_ONLY" = 1 ] && { say "--reset-only: stopping here"; exit 0; }

# --- run ---------------------------------------------------------------------
# The prompt must be byte-identical across arms. Keep it in one place.
PROMPT_FILE="$PROJECT/../java-faker-task-prompt.txt"
if [ ! -f "$PROMPT_FILE" ]; then
  cat > "$PROMPT_FILE" <<'TASK'
Мы будем писать фэйки для существующего java-репозитория, который не
компилируется из-за их отсутствия. Файлы находятся в репозитории
gitlab.mikhailov.tech - серия ca2 и edo-. Не используй ветки fakes оттуда,
пиши и тестируй сам.
TASK
  say "created $PROMPT_FILE — ADD YOUR GITLAB CREDENTIAL HANDLING, then rerun"
  say "(the token is deliberately NOT written into this file)"
  exit 3
fi

START=$(date +%s)
say "launching harness on $ARM ..."
( cd "$PROJECT" && npx --yes @deepseek-ai/dsh --profile headless "$(cat "$PROMPT_FILE")" ) \
  > "$HOME/remeasure-$ARM-$(date +%H%M%S).out" 2>&1
END=$(date +%s)
say "run finished in $(( (END-START)/60 )) min"

# --- milestones --------------------------------------------------------------
say "milestones from the session transcript:"
SESS=$(find "$HOME/.dsh/sessions" -name session.jsonl.zstd -newermt "@$START" | head -1)
[ -n "$SESS" ] && zstd -dc "$SESS" 2>/dev/null | python3 -c "
import sys, json, re
first=None; seen={}; tools=0
for ln in sys.stdin:
    try: d=json.loads(ln)
    except Exception: continue
    t=d.get('time')
    if t and first is None: first=t
    if d.get('type')=='tool/call': tools+=1
    for pat,name in ((r'install:install-file','first artifact installed'),
                     (r'BUILD SUCCESS\b','first maven BUILD SUCCESS'),
                     (r'BUILD SUCCESSFUL','first gradle BUILD SUCCESSFUL')):
        if re.search(pat, ln) and name not in seen and t and first:
            seen[name]=(t-first)/60000.0
print(f'    tool calls: {tools}')
for k,v in sorted(seen.items(), key=lambda x:x[1]): print(f'    {k:28} +{v:5.1f} min')
if not seen: print('    (no milestones reached)')
"
