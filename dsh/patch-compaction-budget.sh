#!/usr/bin/env bash
# Give dsh's compaction summarizer a real output budget.
#
# WHY: in web dsh, compaction is owned by the AGENT PRESET, not the host plane.
# Each preset's agent.cordis.yml mounts `compaction-basic` inside an isolated
# `compaction` group with NO config, so the summarizer runs on plugin defaults:
# maxTokens 8192. A 150-190k-token, 27-tool session summarizes to more than
# 8192 tokens -> finish=output_limit -> finishError rejects it -> compaction
# fails -> retried every turn (~2 min each; 59 laps in 6h). The top-level
# `compaction-basic:` block in ~/.dsh/settings.yaml configures the HOST-plane
# instance, which dsh-web-app/cordis.patch.yml disables - so it never reaches
# the summarizer that actually runs. This patch adds `config:` to the preset row.
#
# RE-RUN AFTER EVERY dsh UPGRADE/REINSTALL: the shipped presets live in the npx
# cache and a reinstall restores them to stock. Idempotent.
#
#   patch-compaction-budget.sh            apply
#   patch-compaction-budget.sh --check    report status only
set -uo pipefail
MODE=apply; [ "${1:-}" = "--check" ] && MODE=check

MAX_TOKENS="${DSH_COMPACTION_MAX_TOKENS:-32768}"
RETAIN_TOKENS="${DSH_COMPACTION_RETAIN_TOKENS:-32768}"
THRESHOLD="${DSH_COMPACTION_THRESHOLD_RATIO:-0.8}"

say() { printf '%s\n' "$*"; }

LIST=$(mktemp -t dsh-preset-targets)
{
  find "$HOME/.npm/_npx" "$HOME/.nvm" /usr/local/lib/node_modules \
    -path '*/@deepseek-ai/dsh-agent-presets/presets/*/agent.cordis.yml' 2>/dev/null
  find "$HOME/.dsh/.agent-presets" -name 'agent.cordis.yml' 2>/dev/null
} | sort -u > "$LIST"

[ -s "$LIST" ] || { say "no agent.cordis.yml presets found" >&2; exit 1; }

rc=0
while IFS= read -r F; do
  [ -n "$F" ] || continue
  grep -q "name: '@deepseek-ai/dsh-compaction-basic'" "$F" || continue
  if grep -qE '^[[:space:]]*# frogs-eye compaction-budget patch$' "$F"; then
    say "patched : $F"; continue
  fi
  if [ "$MODE" = check ]; then say "stock   : $F (summarizer maxTokens = 8192 default)"; continue; fi
  cp "$F" "$F.orig-compaction" 2>/dev/null || true
  F="$F" MAX_TOKENS="$MAX_TOKENS" RETAIN_TOKENS="$RETAIN_TOKENS" THRESHOLD="$THRESHOLD" python3 - <<'PY' || rc=1
import os, re
p = os.environ["F"]
s = open(p).read()
anchor = re.compile(r"^( +)- id: compaction-basic\n( +)name: '@deepseek-ai/dsh-compaction-basic'\n", re.M)
m = anchor.search(s)
if not m:
    raise SystemExit("anchor not found: " + p)
ind = m.group(2)                      # indentation of the `name:` line
block = (m.group(0)
         + f"{ind}# frogs-eye compaction-budget patch\n"
         + f"{ind}config:\n"
         + f"{ind}  maxTokens: {os.environ['MAX_TOKENS']}\n"
         + f"{ind}  retainTokens: {os.environ['RETAIN_TOKENS']}\n"
         + f"{ind}  thresholdRatio: {os.environ['THRESHOLD']}\n")
s = s[:m.start()] + block + s[m.end():]
open(p, "w").write(s)
print("patched :", p)
PY
done < "$LIST"
rm -f "$LIST"

[ "$MODE" = apply ] && say "restart dsh for the change to load (presets are read at session start)."
exit $rc
