#!/usr/bin/env bash
# Bound the number of CONCURRENT subagents a dsh parent may hold, by QUEUEING
# the excess rather than rejecting it.
#
# WHAT THIS DOES
#
#   DSH_MAX_PARALLEL_SUBAGENTS=1   ten subagents still run -- one at a time
#   DSH_MAX_PARALLEL_SUBAGENTS=3   at most three in flight, rest wait their turn
#   DSH_MAX_PARALLEL_SUBAGENTS=0   cap disabled entirely
#
# The spawn call blocks until a sibling settles, so no work is lost and the model
# never has to notice an error and retry. That is deliberate: a REJECTING limiter
# is what LiteLLM's max_parallel_requests does, and it produced 429 storms rather
# than serialization. Same lesson applies here.
#
#   DSH_SUBAGENT_QUEUE_MODE=reject        restore the old throw-on-limit behaviour
#   DSH_SUBAGENT_QUEUE_TIMEOUT_MS=31536000000 safety valve; throws if nothing frees (default 1 year = effectively never)
#
# WHY THIS IS A CODE PATCH AND NOT CONFIGURATION.
#
# Up to 0.1.5 dsh ships no breadth limit on delegation (verified on dsh-subagent
# 0.1.1-rc.2 and again on 0.1.5-rc.2, which reads no env vars at all and whose
# admission gate only guards teardown draining). Later versions (0.1.7, 0.2.0)
# have a native `subagent.maxActiveSubagents` (default 8) that REJECTS a spawn
# beyond it; this patch still applies there and turns that into a queue at its
# own cap. Before 0.1.7:
#   * `maxDepth` (default 3) bounds RECURSION, not width.
#   * `agent-loop.maxParallelToolCalls` bounds tool calls in flight for one
#     agent step. A background subagent spawn returns immediately and frees the
#     slot, so it never applies -- 13 concurrent children were observed under
#     `maxParallelToolCalls: 1`.
#   * `jobs-local.maxConcurrentJobsPerOwner` bounds background JOBS, and looks
#     like the knob. It is not: with `backgroundMode: continuable` (what the
#     shipped `standard` and `code` presets use) the subagent tool returns from
#     `ctx.subagents.startContinuable()` BEFORE reaching the job registry.
#
# Every configuration route to this limit works by editing the agent preset, and
# editing the preset changes the system prompt the model receives. On this estate
# that mattered: sessions whose prompt had been altered dropped into a degenerate
# `!` repetition loop that ran to the 32,768-token cap and produced nothing --
# twice, in two sessions. Cause never proven; the cheapest way to stop paying to
# find out is to stop changing the prompt.
#
# This patch changes no prompt text at all. The model sees the stock system
# prompt, the stock tool schema, and the stock tool descriptions.
#
# TWO GENERATIONS OF dsh INTERNALS ARE SUPPORTED (auto-detected by anchor):
#
#   dsh-subagent >= 0.1.5 (current): the activation registry moved into
#   `this.activations` (ContinuableActivationRegistry) with a `resident` Map;
#   `stateOf()` and `accepted` are gone. A finished continuable child STAYS
#   resident (idle, resumable), so the count keys on `agent.status === "running"`
#   -- counting residency was the slot leak that deadlocked limit>=2 on the old
#   patch ("waited 1800s for a free subagent slot (limit 2) and none freed").
#   `startContinuable` also computes `childId` right after admission, so each
#   reservation is keyed to its OWN child and released exactly when that child
#   materializes: N simultaneous spawns can no longer release each other's slot.
#
#   dsh-subagent 0.1.1-rc.2 (legacy): flat `this.activations` Map + `stateOf()`.
#   Kept as a fallback so pinning back still works.
#
# RE-RUN THIS AFTER EVERY dsh UPGRADE. npx installs into a content-addressed
# cache directory, so a new version lands in a new path with the file restored
# to stock. A bare `npx @deepseek-ai/dsh web` re-resolves versions and can do
# this silently; prefer a pinned global install (`npm i -g @deepseek-ai/dsh@X`)
# so the patched file stays put until YOU choose to upgrade.
#
#   DSH_PATCH_SEARCH_ROOT              extra dir to search (e.g. a local
#                                      install root like /opt/dsh)
#
#   patch-subagent-limit.sh            apply (idempotent)
#   patch-subagent-limit.sh --revert   restore the shipped file
#   patch-subagent-limit.sh --check    report status only
set -uo pipefail

MODE=apply
case "${1:-}" in
  --revert) MODE=revert ;;
  --check)  MODE=check ;;
  "")       ;;
  *) echo "unknown option: $1" >&2; exit 2 ;;
esac

say() { printf '%s\n' "$*"; }

# --- locate the installed manager, newest first -----------------------------
# macOS ships bash 3.2, which has no `mapfile` and no `readarray`; a temp file
# keeps this portable across every shell the estate actually has.
LIST=$(mktemp "${TMPDIR:-/tmp}/dsh-subagent-targets.XXXXXX")
trap 'rm -f "$LIST"' EXIT
find "$HOME/.npm/_npx" "$HOME/.nvm" /usr/local/lib/node_modules ${DSH_PATCH_SEARCH_ROOT:-} \
     /opt/homebrew/lib/node_modules "$HOME/.local/lib/node_modules" \
  -path '*/@deepseek-ai/dsh-subagent/lib/index.js' \
  2>/dev/null | grep -v '\.orig$' > "$LIST" || true

if [ ! -s "$LIST" ]; then
  echo "no @deepseek-ai/dsh-subagent install found" >&2
  exit 1
fi

NODE=$(command -v node || echo "$HOME/.local/bin/node")

while IFS= read -r TARGET; do
  [ -n "$TARGET" ] || continue
  say "target: $TARGET"
  VER=$(sed -nE 's/.*"version": *"([^"]+)".*/\1/p' "$(dirname "$TARGET")/../package.json" 2>/dev/null | head -1)
  [ -n "$VER" ] && say "  dsh-subagent version: $VER"

  if grep -q 'awaitParallelSubagentSlot' "$TARGET" 2>/dev/null; then
    STATE=patched
  elif grep -q 'assertParallelSubagentBudget' "$TARGET" 2>/dev/null; then
    STATE=patched-old-rejecting
  else
    STATE=stock
  fi

  if [ "$MODE" = check ]; then
    say "  state: $STATE"
    continue
  fi

  if [ "$MODE" = revert ]; then
    if [ -f "$TARGET.orig" ]; then
      cp "$TARGET.orig" "$TARGET" && say "  reverted from .orig"
    else
      say "  no .orig to revert from -- leaving as is"
    fi
    continue
  fi

  # apply
  if [ "$STATE" = patched ]; then
    say "  already patched -- nothing to do"
    continue
  fi
  if [ "$STATE" = patched-old-rejecting ]; then
    if [ -f "$TARGET.orig" ]; then
      say "  superseding the old rejecting patch"
      cp "$TARGET.orig" "$TARGET"
    else
      say "  ABORT: old rejecting patch present but no .orig to rebuild from"
      continue
    fi
  fi
  # A stale .orig from a PREVIOUS dsh version must not be reused: npx overwrites
  # index.js in place on reinstall, leaving an .orig that no longer matches.
  # The current file is stock at this point, so refresh the backup from it.
  cp "$TARGET" "$TARGET.orig"

  TARGET="$TARGET" "$NODE" - <<'JS'
const fs = require('fs');
const p = process.env.TARGET;
let s = fs.readFileSync(p, 'utf8');

// ---------------------------------------------------------------------------
// Generation 2: dsh-subagent >= 0.1.5 (ContinuableActivationRegistry + resident)
// ---------------------------------------------------------------------------
const newHead = `\tasync startContinuable(spec) {
\t\tconst request = spec.request;
\t\tconst parent = request.parent;
\t\tthis.activations.assertAdmitting(parent);`;

const newInject = `\t\tconst childId = spec.childId ?? brandString(randomUUID());
\t\tthis.activations.assertChildIdAvailable(childId);`;

// coldResume() is the OTHER way a child becomes resident (resuming a persisted
// continuable child when a follow-up arrives after it was disposed). It calls
// materialize() too, so it must be gated or a resume can exceed the limit.
// The authorizeLineage(...source.header.parentSession) line is unique to it.
const newColdResume = `\t\t\tthis.activations.assertAdmitting(parent);
\t\t\tthis.activations.authorizeLineage(parent, childId, source.header.parentSession);`;

// ---------------------------------------------------------------------------
// Generation 1: dsh-subagent 0.1.1-rc.2 (flat activations Map + stateOf)
// ---------------------------------------------------------------------------
const oldAnchor = `\tasync startContinuable(spec) {
\t\tconst request = spec.request;
\t\tconst parent = request.parent;
\t\tthis.assertAdmitting(parent);`;

const oldHelperAnchor = `\t/** Reject new admission once the manager or this exact parent tree began draining. */
\tassertAdmitting(agent) {`;

// The queueing gate is identical across generations; only the reserve signature
// differs (gen 2 passes the child's id so the reservation is per-child).
// Shared by both generations. `sig` is the reserve signature; `who` is the
// child id expression to name in debug lines (gen 2 has one, gen 1 does not).
const gateBody = (sig, who) => `\t/** frog's-eye patch: stderr trace, on only when DSH_SUBAGENT_LIMIT_DEBUG=1. */
\t_feLog(message) {
\t\tif (process.env.DSH_SUBAGENT_LIMIT_DEBUG === "1") process.stderr.write("[subagent-limit] " + message + "\\n");
\t}
\t/**
\t * frog's-eye patch: bound CONCURRENT direct continuable children per parent by
\t * QUEUEING the excess. All requested subagents still run; only their overlap is
\t * bounded. See dsh/patch-subagent-limit.sh in the frogs-eye repo for why this is
\t * code and not configuration.
\t *   DSH_MAX_PARALLEL_SUBAGENTS   cap (default 1, 0 disables)
\t *   DSH_SUBAGENT_QUEUE_MODE      "queue" (default) or "reject"
\t *   DSH_SUBAGENT_QUEUE_TIMEOUT_MS safety valve, default 31536000000 (1 year)
\t *   DSH_SUBAGENT_LIMIT_DEBUG=1   trace admit/queue/release with live counts to stderr
\t */
\tasync awaitParallelSubagentSlot(${sig}) {
\t\tconst rawLimit = process.env.DSH_MAX_PARALLEL_SUBAGENTS;
\t\tconst limit = rawLimit === undefined || rawLimit === "" ? 1 : Number(rawLimit);
\t\tif (!Number.isFinite(limit) || limit <= 0) return;
\t\tlet live = this.countLiveSubagents(parent);
\t\tif (live < limit) { this._feReserveSlot(${sig}); this._feLog(\`admit parent=\${parent.id} child=\${${who}} live=\${live + 1}/\${limit}\`); return; }
\t\tconst mode = String(process.env.DSH_SUBAGENT_QUEUE_MODE || "queue").toLowerCase();
\t\tif (mode === "reject") {
\t\t\tthrow new SubagentError(\`parallel subagent limit reached (limit: \${limit}) — wait for a running subagent to finish before starting another, then retry\`, "PARALLEL_LIMIT");
\t\t}
\t\tthis._feLog(\`queue parent=\${parent.id} child=\${${who}} live=\${live}/\${limit}\`);
\t\tconst rawTimeout = process.env.DSH_SUBAGENT_QUEUE_TIMEOUT_MS;
\t\tconst parsedTimeout = rawTimeout === undefined || rawTimeout === "" ? 31536000000 : Number(rawTimeout);
\t\tconst timeoutMs = Number.isFinite(parsedTimeout) && parsedTimeout > 0 ? parsedTimeout : 31536000000;
\t\tconst deadline = Date.now() + timeoutMs;
\t\tconst started = Date.now();
\t\tfor (;;) {
\t\t\tawait new Promise((resolve) => setTimeout(resolve, 250));
\t\t\tlive = this.countLiveSubagents(parent);
\t\t\tif (live < limit) { this._feReserveSlot(${sig}); this._feLog(\`admit-after-wait parent=\${parent.id} child=\${${who}} waited=\${Math.round((Date.now() - started) / 1000)}s live=\${live + 1}/\${limit}\`); return; }
\t\t\tif (Date.now() >= deadline) {
\t\t\t\tthis._feLog(\`timeout parent=\${parent.id} child=\${${who}} live=\${live}/\${limit}\`);
\t\t\t\tthrow new SubagentError(\`waited \${Math.round(timeoutMs / 1000)}s for a free subagent slot (limit \${limit}) and none freed\`, "PARALLEL_QUEUE_TIMEOUT");
\t\t\t}
\t\t}
\t}
`;

if (s.includes(newHead) && s.includes(newInject)) {
  if (!s.includes(newColdResume)) {
    console.log('  ABORT: coldResume anchor not found -- gating only startContinuable would leave the');
    console.log('  resume path ungated (a resumed child could exceed the limit). Update the anchor first.');
    process.exit(3);
  }
  const helpers = `\t/**
\t * frog's-eye patch (dsh-subagent >= 0.1.5): direct children of \`parent\` that are
\t * ACTIVELY RUNNING a turn, plus slots granted to children not yet materialized.
\t * Counts only status === "running": a finished continuable child stays resident
\t * (idle, resumable) and must not hold a slot -- counting residency was the slot
\t * leak that deadlocked limit>=2 ("waited 1800s ... and none freed").
\t */
\tcountLiveSubagents(parent) {
\t\tlet live = 0;
\t\tfor (const activation of this.activations.resident.values()) {
\t\t\tif (activation.parentSession !== parent.id) continue;
\t\t\tif (activation.inbox.closing !== void 0) continue;
\t\t\tif (activation.handle.agent.status !== "running") continue;
\t\t\tlive++;
\t\t}
\t\treturn live + ((this._feReservedSlots && this._feReservedSlots.get(parent.id)) || 0);
\t}
\t/**
\t * Hold a slot for ONE specific child from grant until that child materializes in
\t * this.activations.resident (60s failsafe if the spawn dies first). Keyed on the
\t * child's own id, so N simultaneous spawns each release exactly their own
\t * reservation -- no cross-release race. Callers reserve with NO await between the
\t * capacity check and this call, which makes the pair atomic on the event loop.
\t */
\t_feReserveSlot(parent, childId) {
\t\tif (!this._feReservedSlots) this._feReservedSlots = new Map();
\t\tconst slots = this._feReservedSlots;
\t\tslots.set(parent.id, (slots.get(parent.id) || 0) + 1);
\t\tvoid (async () => {
\t\t\tconst deadline = Date.now() + 6e4;
\t\t\tfor (;;) {
\t\t\t\tawait new Promise((resolve) => setTimeout(resolve, 50));
\t\t\t\tif (this.activations.resident.has(childId) || Date.now() >= deadline) break;
\t\t\t}
\t\t\tconst current = slots.get(parent.id) || 0;
\t\t\tif (current <= 1) slots.delete(parent.id); else slots.set(parent.id, current - 1);
\t\t\tthis._feLog(\`release-reservation parent=\${parent.id} child=\${childId} materialized=\${this.activations.resident.has(childId)}\`);
\t\t})();
\t}
` + gateBody('parent, childId', 'childId');
  s = s.replace(newHead, helpers + newHead);
  s = s.replace(newInject, newInject + '\n\t\tawait this.awaitParallelSubagentSlot(parent, childId);');
  s = s.replace(newColdResume, newColdResume + '\n\t\t\tawait this.awaitParallelSubagentSlot(parent, childId);');
  fs.writeFileSync(p, s);
  console.log('  patched (queueing mode, generation 2: resident/status model; startContinuable + coldResume gated)');
  process.exit(0);
}

if (s.includes(oldAnchor) && s.includes(oldHelperAnchor)) {
  const helpers = `\t/**
\t * frog's-eye patch (legacy dsh-subagent 0.1.1-rc.2): live direct continuable
\t * children of one parent, PLUS slots already granted whose Activation has not
\t * registered yet (counting only this.activations races the async spawn path).
\t */
\tcountLiveSubagents(parent) {
\t\tlet live = 0;
\t\tfor (const activation of this.activations.values()) {
\t\t\tif (activation.parentSession !== parent.id) continue;
\t\t\tif (this.stateOf(activation) === "settled") continue;
\t\t\tlive++;
\t\t}
\t\treturn live + ((this._feReservedSlots && this._feReservedSlots.get(parent.id)) || 0);
\t}
\t_feReserveSlot(parent) {
\t\tif (!this._feReservedSlots) this._feReservedSlots = new Map();
\t\tconst slots = this._feReservedSlots;
\t\tslots.set(parent.id, (slots.get(parent.id) || 0) + 1);
\t\tconst before = new Set();
\t\tfor (const activation of this.activations.values()) {
\t\t\tif (activation.parentSession === parent.id) before.add(activation.childId);
\t\t}
\t\tvoid (async () => {
\t\t\tconst deadline = Date.now() + 6e4;
\t\t\tfor (;;) {
\t\t\t\tawait new Promise((resolve) => setTimeout(resolve, 50));
\t\t\t\tlet appeared = false;
\t\t\t\tfor (const activation of this.activations.values()) {
\t\t\t\t\tif (activation.parentSession !== parent.id) continue;
\t\t\t\t\tif (!before.has(activation.childId)) { appeared = true; break; }
\t\t\t\t}
\t\t\t\tif (appeared || Date.now() >= deadline) break;
\t\t\t}
\t\t\tconst current = slots.get(parent.id) || 0;
\t\t\tif (current <= 1) slots.delete(parent.id); else slots.set(parent.id, current - 1);
\t\t})();
\t}
` + gateBody('parent', '"?"');
  s = s.replace(oldHelperAnchor, helpers + oldHelperAnchor);
  s = s.replace(oldAnchor, oldAnchor + '\n\t\tawait this.awaitParallelSubagentSlot(parent);');
  fs.writeFileSync(p, s);
  console.log('  patched (queueing mode, generation 1: legacy stateOf model)');
  process.exit(0);
}

console.log('  ABORT: no known startContinuable anchor found -- dsh internals changed again.');
console.log('  Inspect the file and add a new anchor generation to this patch before trusting the limit.');
process.exit(3);
JS
  rc=$?
  [ "$rc" -ne 0 ] && { say "  patch failed (rc=$rc) -- restoring shipped file"; cp "$TARGET.orig" "$TARGET"; continue; }

  if "$NODE" --check "$TARGET" 2>/dev/null; then
    say "  syntax OK"
  else
    say "  SYNTAX ERROR after patch -- restoring shipped file"
    cp "$TARGET.orig" "$TARGET"
  fi
done < "$LIST"

say ""
say "restart dsh for the change to load (the module is read at process start)."
