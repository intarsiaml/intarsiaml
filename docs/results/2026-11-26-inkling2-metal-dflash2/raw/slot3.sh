#!/bin/bash
# endor EXCLUSIVE fleetq slot 3 / 4, lane ink-metal-dflash2: ABBA halves at one k (K, default 5): ORDER=a "dspark dfl dfl dspark" reps R1 R2,
# ORDER=b "dfl dspark dspark dfl" reps R1 R2; speed.json, 256 tokens, EOS honoured, 45 s idle before every process (slot.sh). With TESTS=1
# first the bf16 draft-pass test of this lane on the Metal backend and the Metal dyn_conv tests (the 2304 shape).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
trap 'for c in $(pgrep -P $$ 2>/dev/null); do pkill -P $c 2>/dev/null; kill $c 2>/dev/null; done' EXIT
W=/opt/code/intarsiaml/.worktrees/ink-metal-dflash2; R=$W/docs/results/2026-11-26-inkling2-metal-dflash2/raw
cd $W
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
K=${K:-5}; R1=${R1:-r1}; R2=${R2:-r2}
echo "slot start $(date +%T) K $K order ${ORDER:-a} $R1 $R2"
if [ -n "${TESTS:-}" ]; then
  mkdir -p .scratch/metal/s3
  env INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2_dflash2 draft_pass -- --nocapture > .scratch/metal/s3/draftpass-metal.txt 2>&1
  echo "== bf16 draft pass on metal rc $? | $(grep -aE '^test |test result' .scratch/metal/s3/draftpass-metal.txt | tr '\n' ' ')"
  cargo test --release -p intarsia-backend-metal --test dflash2 > .scratch/metal/s3/metal-dflash2-tests.txt 2>&1
  echo "== metal backend dflash2 tests rc $? | $(grep -aE '^test result' .scratch/metal/s3/metal-dflash2-tests.txt | tr '\n' ' ')"
  # the binary is the merged head (#544 in): greedy ids again, 256 tokens (md5.json, --ignore-eos), off / default / k$K
  NAME=s3-md5 PROMPTS=md5 TOK=256 IDLE=0 LIMIT=300 ARMS="off@r1 dspark@r1 dfl$K@r1" $R/slot.sh
fi
if [ "${ORDER:-a}" = a ]; then A="dspark@$R1 dfl$K@$R1 dfl$K@$R2 dspark@$R2"; else A="dfl$K@$R1 dspark@$R1 dspark@$R2 dfl$K@$R2"; fi
NAME=${NAME:?} PROMPTS=speed TOK=256 IDLE=45 LIMIT=${LIMIT:-1000} ARMS="$A" $R/slot.sh
echo "slot end $(date +%T)"
