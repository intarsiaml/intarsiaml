#!/usr/bin/env bash
# endor EXCLUSIVE fleetq slot (<= 15 min), lane ink-hip-specflip, gate 3 (the Metal no-op check: the default rule is shared code).
#   MAIN=<bin name> BR=<bin name> bash metal_slot.sh   (binaries .scratch/bin/intarsia-<name>, built with --features metal)
# 1. greedy md5, md5.json (golden `code` 101 ids + the 4108-id prompt), 256 tokens, --ignore-eos: main vs the branch, --spec off and the
#    default (Metal: DSpark k3); 2. one default sampled `generate` on the branch (T 1.0): the startup line must say drafting off (Metal
#    stays opt-in); 3. the Metal tiny suites of the sampled / DSpark paths (test executables of the branch head, target/release/deps).
# No INTARSIA_* variable set. Kills only its own children on exit.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done; kill "$1" 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree "$c"; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset "$v"; done
W=/opt/code/intarsiaml/.worktrees/ink-hip-specflip; S=$W/.scratch; RAW=$W/docs/results/2026-11-26-inkling2-hip-specflip/raw
MAIN=${MAIN:?}; BR=${BR:?}
O=$RAW/metal-$BR; mkdir -p "$O"
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate)' | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
echo "[metal] start $(date +%T) | $MAIN $(md5 -q "$S/bin/intarsia-$MAIN") $(cat "$S/bin/intarsia-$MAIN.commit" 2>/dev/null) | $BR $(md5 -q "$S/bin/intarsia-$BR") $(cat "$S/bin/intarsia-$BR.commit" 2>/dev/null)"
for b in "$MAIN" "$BR"; do for sp in off def; do
  x=""; [ $sp = off ] && x="--spec off"
  f=$O/md5-$b-$sp
  "$S/bin/intarsia-$b" generate -m "$M" --backend metal --max-seq 8192 --profile-every 0 --show-ids --ignore-eos --max-tokens 256 --prompt-ids "$RAW/md5.json" $x > "$f.out" 2> "$f.err"
  echo "[metal] md5 $b $sp rc $? $(date +%T) ids-md5 $(cat "$f.out" "$f.err" | grep -a '^ids\[' | md5 -q) lines $(cat "$f.out" "$f.err" | grep -ac '^ids\[') | $(grep -aE '^spec: verify' "$f.err" | cut -c1-110)"
done; done
f=$O/sampled-def
"$S/bin/intarsia-$BR" generate -m "$M" --backend metal --max-tokens 64 --temperature 1.0 --seed 1 --prompt "Write a short story about a lighthouse." > "$f.out" 2> "$f.err"
echo "[metal] sampled default rc $? $(date +%T) | $(grep -aE 'DFlash-family drafting|draft length per request|^spec: verify|^decode overall' "$f.err" | tr '\n' ' ' | cut -c1-400)"
for t in inkling2_sampled_default inkling2_dspark_sampled inkling2_dspark_sampled_adapt inkling2_dspark; do
  exe=$(ls -t "$W"/target/release/deps/$t-* 2>/dev/null | grep -v '\.d$' | head -1)
  [ -z "$exe" ] && { echo "[metal] tiny $t: no executable"; continue; }
  env INTARSIA_INKLING2_BACKEND=metal "$exe" --test-threads=1 > "$O/tiny-$t.txt" 2>&1
  echo "[metal] tiny $t (metal) rc $? $(date +%T) | $(grep -aE '^test result' "$O/tiny-$t.txt") | $(grep -aE ' FAILED|panicked' "$O/tiny-$t.txt" | head -2 | tr '\n' ' ')"
done
echo "[metal] done $(date +%T)"
