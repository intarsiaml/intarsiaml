#!/bin/bash
# lane ink-decroof job A (exclusive, quiet): CONV_PRE speed, --spec off, 256 greedy tokens, --ignore-eos.
#  1. within-process: the diag binary, 84 prompts per process (7 cycles x 143 / 1k / 4k x arms A all off / B CONV_PRE /
#     C + BIAS_PRE / D + BIAS_EARLY, ABCD / DCBA by cycle),
#     PROCS processes
#  2. cross-process ABBA: base and cand binaries, NQ processes per arm in the order A B B A ..., each the three prompts once
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'echo "CUT (TERM) at step ${STEP:-?}"; cleanup; exit 143' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-decroof; S=$W/.scratch; O=$S/${OUT:-A}; mkdir -p $O
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
BASE=$S/bin/intarsia-base; CAND=$S/bin/intarsia-cand; DIAG=$S/bin/intarsia-diag
PROCS=${PROCS:-3}; NQ=${NQ:-6}
for f in $BASE $CAND $DIAG; do [ -x $f ] || { echo "ABORT: missing $f"; exit 6; }; done
cd $W
echo "jobA start $(date +%T) base $(md5 -q $BASE) cand $(md5 -q $CAND) diag $(md5 -q $DIAG) procs $PROCS nq $NQ"
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate|accuracy)' || true)
[ -n "$others" ] && { echo "ABORT: other engine processes on the box:"; echo "$others"; exit 5; }
G="--model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --spec off --max-tokens 256 --ignore-eos"
STEP=within
ARMS=$(cat $S/arms_ab.txt)
for p in $(seq 1 $PROCS); do
  t0=$(date +%s)
  env INTARSIA_DIAG_ARMS=$ARMS $DIAG generate $G --prompt-ids $S/prompts_ab.json > $O/p$p.out 2> $O/p$p.err
  echo "within p$p exit $? $(( $(date +%s) - t0 )) s, launch counts (conv_pre bias_pre bias_early) per prompt: $(grep -h '^cpre\[' $O/p$p.err | awk '{print $2 "/" $3 "/" $4}' | sort | uniq -c | tr '\n' ' ')"
done
python3 -I $S/ab_within.py $O
STEP=cross
i=0
for r in $(seq 1 $NQ); do
  if [ $((r % 2)) = 1 ]; then order="base cand"; else order="cand base"; fi
  for a in $order; do
    i=$((i + 1)); bin=$S/bin/intarsia-$a; t0=$(date +%s)
    env X=1 $bin generate $G --prompt-ids $S/prompts.json > $O/q${i}_$a.out 2> $O/q${i}_$a.err
    echo "cross q$i $a exit $? $(( $(date +%s) - t0 )) s | $(grep -h '^decode\[' $O/q${i}_$a.err | sed 's/decode//;s/ tokens, [0-9]* ms//' | tr '\n' ' ')"
  done
done
python3 -I $S/ab_proc.py $O
echo "jobA done $(date +%T)"
