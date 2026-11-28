#!/bin/bash
# endor EXCLUSIVE fleetq slot D, lane ink-dflash-cut: why cut7 was slower than dfl5 in slot A rep 1 (37.4 vs 33.9 ms a pass at fewer
# rows). Three traced k7 arms over speed.json (INTARSIA_SPEC_TRACE=1: per pass draft phases + "verify + rollback" ms), 45 s idle before each:
#   dfl7  = k7, no cut;  lazy7 = k7 + INTARSIA_DFLASH_CUT=1e-30 (the cut's row-by-row host path, every draft kept);  cut7 = k7 + cut 0.3.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-dflash-cut; R=$W/docs/results/2026-11-27-inkling2-dflash-cut/raw; O=$R/D; mkdir -p $O
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
D=$HOME/.cache/huggingface/hub/models--incoai--Inkling-2-Mini-DFlash2/snapshots/4f9088ea66c21f2b99ed657d1941c306b03e84d2
B=$W/.scratch/bin/intarsia-ct
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate)' | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
echo "slot D start $(date +%T) binary md5 $(md5 -q $B) git $(git -C $W rev-parse --short HEAD)"
G="generate -m $M --backend metal --max-seq 20480 --profile-every 0 --show-ids --max-tokens 256 --prompt-ids $R/speed.json --spec dflash --spec-draft $D --spec-k 7"
for arm in dfl7 lazy7 cut7; do
  case $arm in dfl7) E="" ;; lazy7) E="INTARSIA_DFLASH_CUT=1e-30" ;; cut7) E="INTARSIA_DFLASH_CUT=0.3" ;; esac
  sleep 45
  env INTARSIA_SPEC_TRACE=1 $E $B $G > $O/$arm.out 2> $O/$arm.err
  echo "$arm rc $? $(date +%T) | $(grep -aoE 'dflash: cut [0-9.e-]+' $O/$arm.err | head -1) | $(grep -aE '^decode overall' $O/$arm.err) | $(grep -aE '^spec: verify' $O/$arm.err)"
done
echo "slot D end $(date +%T)"
