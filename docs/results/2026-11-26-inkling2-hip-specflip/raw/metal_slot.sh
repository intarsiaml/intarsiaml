#!/usr/bin/env bash
# endor EXCLUSIVE fleetq slot (<= 15 min), lane ink-hip-specflip, gate 3 (the Metal no-op check: the default rule is shared code).
#   MAIN=<bin name> BR="<bin name> [<bin name> ...]" bash metal_slot.sh   (binaries .scratch/bin/intarsia-<name>, built with --features metal;
#   tiny test executables .scratch/bin/mt-<name>-<test>, built from the same tree; v2: several branch heads in one slot)
# 1. greedy md5, md5.json (golden `code` 101 ids + the 4108-id prompt), 256 tokens, --ignore-eos: main vs the branch, --spec off and the
#    default (Metal: DSpark k3); 2. one default sampled `generate` on the branch (T 1.0): the startup line must say drafting off (Metal
#    stays opt-in); 3. the Metal tiny suites of the sampled / DSpark paths (test executables of the branch head, target/release/deps).
# No INTARSIA_* variable set. Kills only its own children on exit.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done; kill "$1" 2>/dev/null; }
LINK=""  # set when this slot made the drafter symlink beside the checkpoint; removed on exit (the link only)
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree "$c"; done; [ -n "$LINK" ] && [ -L "$LINK" ] && rm "$LINK" && echo "[metal] removed the drafter link $LINK"; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset "$v"; done
W=/opt/code/intarsiaml/.worktrees/ink-hip-specflip; S=$W/.scratch; RAW=$(cd "$(dirname "$0")" && pwd)  # the raw dir this copy sits in
MAIN=${MAIN:?}; BR=${BR:?}
O=$RAW/metal-${TAG:-$(echo $BR | tr ' ' '-')}; mkdir -p "$O"
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
# (only processes whose command starts with an intarsia binary: a shell whose command line mentions one is not a run)
others=$(pgrep -fl '^[^ ]*/intarsia[^ /]* (serve|generate)' | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
echo "[metal] start $(date +%T)"
T0=$(date +%s); CAP=${CAP:-1080}  # the slot stops its own work at CAP s (default 18 min) so it ends inside the <= 20 min grant
for b in $MAIN $BR; do [ -x "$S/bin/intarsia-$b" ] || { echo "GUARD: no binary $S/bin/intarsia-$b"; exit 2; }; done
# run_capped NAME CMD...: runs CMD in the background, kills its tree if the slot's CAP is reached (rc 124)
run_capped() { local n=$1; shift; "$@" & local p=$!; while kill -0 $p 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -ge $CAP ] && { echo "[metal] CAP reached during $n: stopped"; kill_tree $p; wait $p 2>/dev/null; return 124; }; sleep 2; done; wait $p; }
# (a) (coordinator 01:14 item 3): the DFlash2 drafter sits next to the checkpoint on endor; the default must still load DSpark k3 alone
# endor keeps the drafter (fleetq #12464's copy) in its own HF cache entry; DRAFT_SRC=<that dir> makes a symlink named as the recipe's
# dflash2_draft beside the checkpoint for this slot only (no folder is made; the link is removed on exit)
D=$M/../Inkling-2-Mini-DFlash2
if [ -n "${DRAFT_SRC:-}" ] && [ ! -e "$D" ]; then ln -s "$DRAFT_SRC" "$D" && LINK=$D && echo "[metal] linked $D -> $DRAFT_SRC"; fi
echo "[metal] DFlash2 drafter beside the checkpoint: $(ls -d "$M/../Inkling-2-Mini-DFlash2" 2>/dev/null || echo absent) ($(ls "$M/../Inkling-2-Mini-DFlash2" 2>/dev/null | tr '\n' ' ' | cut -c1-120))"
for b in $MAIN $BR; do echo "[metal] bin $b $(md5 -q "$S/bin/intarsia-$b") $(cat "$S/bin/intarsia-$b.commit" 2>/dev/null)"; done
for b in $MAIN $BR; do sleep ${GAP:-45}; for sp in off def; do
  x=""; [ $sp = off ] && x="--spec off"
  f=$O/md5-$b-$sp
  "$S/bin/intarsia-$b" generate -m "$M" --backend metal --max-seq 8192 --profile-every 0 --show-ids --ignore-eos --max-tokens 256 --prompt-ids "$RAW/md5.json" $x > "$f.out" 2> "$f.err"
  echo "[metal] md5 $b $sp rc $? $(date +%T) ids-md5 $(cat "$f.out" "$f.err" | grep -a '^ids\[' | md5 -q) lines $(cat "$f.out" "$f.err" | grep -ac '^ids\[') | $(grep -aE '^spec: verify' "$f.err" | cut -c1-110)"
  [ $sp = def ] && echo "[metal]   $b default load: $(grep -aE 'DSpark default|DFlash2|second draft|^dflash: ' "$f.err" | tr '\n' ' ' | cut -c1-400)"
done; done
for b in $BR; do
  f=$O/sampled-def-$b
  "$S/bin/intarsia-$b" generate -m "$M" --backend metal --max-tokens 64 --temperature 1.0 --seed 1 --prompt "Write a short story about a lighthouse." > "$f.out" 2> "$f.err"
  echo "[metal] $b sampled default rc $? $(date +%T) | $(grep -aE 'DFlash-family drafting|draft length per request|DFlash2 default|^--spec dflash|^spec: verify|^decode overall' "$f.err" | tr '\n' ' ' | cut -c1-500)"
  sleep ${GAP:-45}  # endor: an idle gap before the tiny suites (nothing in this slot is timed; gaps sit between run groups)
  for t in inkling2_sampled_default inkling2_dspark_sampled inkling2_dspark_sampled_adapt inkling2_dspark inkling2_dflash2; do
    exe=$S/bin/mt-$b-$t
    [ -x "$exe" ] || { echo "[metal] $b tiny $t: no executable"; continue; }
    [ $(( $(date +%s) - T0 )) -ge $CAP ] && { echo "[metal] $b tiny $t: skipped (CAP)"; continue; }
    run_capped "tiny $t" env INTARSIA_INKLING2_BACKEND=metal "$exe" --test-threads=1 > "$O/tiny-$b-$t.txt" 2>&1
    echo "[metal] $b tiny $t (metal) rc $? $(date +%T) | $(grep -aE '^test result' "$O/tiny-$b-$t.txt") | $(grep -aE ' FAILED|panicked' "$O/tiny-$b-$t.txt" | head -2 | tr '\n' ' ')"
  done
done
echo "[metal] done $(date +%T)"
