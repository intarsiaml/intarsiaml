#!/bin/bash
# endor EXCLUSIVE fleetq slot T, lane ink-dflash-cut: confidence traces for the offline cut simulation. generate over speed.json (the
# 19 prompts of #546's k sweep, 256 tokens, EOS honoured), greedy, DFlash2 k7 then k5, INTARSIA_SPEC_TRACE=1 + INTARSIA_DFLASH_CONF_TRACE=1.
# Timing is not a measurement here (the traces sync between phases); ids are checked against the untraced k5 md5s.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-dflash-cut; R=$W/docs/results/2026-11-27-inkling2-dflash-cut/raw; O=$R/T; mkdir -p $O
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
D=$HOME/.cache/huggingface/hub/models--incoai--Inkling-2-Mini-DFlash2/snapshots/4f9088ea66c21f2b99ed657d1941c306b03e84d2
B=$W/.scratch/bin/intarsia-ct
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate)' | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
echo "slot T start $(date +%T) binary md5 $(md5 -q $B) git $(git -C $W rev-parse --short HEAD)"
G="generate -m $M --backend metal --max-seq 20480 --profile-every 0 --show-ids --max-tokens 256 --prompt-ids $R/speed.json"
for k in ${KS:-7 5}; do
  f=$O/dfl$k${TAG:-}
  env INTARSIA_SPEC_TRACE=1 INTARSIA_DFLASH_CONF_TRACE=1 $B $G --spec dflash --spec-draft $D --spec-k $k > $f.out 2> $f.err
  echo "dfl$k traced rc $? $(date +%T) | conf lines $(grep -ac 'dflash2 conf' $f.err) | accepted lines $(grep -acE '^spec: .*token [0-9]+ pos [0-9]+ drafted' $f.err) | $(grep -aE '^spec: verify' $f.err)"
done
echo "slot T end $(date +%T)"
