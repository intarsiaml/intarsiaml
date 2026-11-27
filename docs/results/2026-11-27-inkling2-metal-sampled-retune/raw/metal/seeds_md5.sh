#!/bin/bash
# endor EXCLUSIVE fleetq slot (<= 20 min), lane ink-metal-sampled2: clause 3 (seeded ids, real model, seeds 0..100 x T 1.0 / 0.6, 24 tokens, k3: drafting off vs
# the policy with POLSTR) and clause 4 (greedy ids over md5.json, 64 tokens, `--spec off` and the recipe default, BIN = main's binary).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. /opt/code/intarsiaml/.worktrees/ink-metal-sampled2/docs/results/2026-11-27-inkling2-metal-sampled-retune/raw/metal/common.sh
: "${POLSTR:?POLSTR}"
TAG=$1; O=$S/metal/seeds-$TAG; mkdir -p $O; guard
RT=${RT:-$(ls -t $W/target/release/deps/inkling2_sampled_real-* | grep -v '\.d$' | head -1)}
echo "start $(date +%T) out $O | test binary $RT | pol string: $POLSTR"
for a in 0 1; do
  rm -f $O/seeds-$a.txt
  TR=INTARSIA_SPEC_TRACE=0; [ $a = 1 ] && TR=INTARSIA_SPEC_TRACE=1
  env INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_INKLING2_BACKEND=metal INTARSIA_SPEC_SAMPLED_DFLASH=$a INTARSIA_SPEC_SAMPLED_ADAPT=$POLSTR \
    $TR INTARSIA_SS_SEEDS=0..100 INTARSIA_SS_TEMPS=1.0,0.6 INTARSIA_SS_TOKENS=24 INTARSIA_SS_K=3 INTARSIA_SS_BUDGET_S=${SS_BUDGET:-400} \
    INTARSIA_SS_OUT=$O/seeds-$a.txt perl -e 'alarm shift; exec @ARGV' 700 $RT --nocapture > $O/seeds-$a.log 2>&1
  echo "seeds arm $a rc $? $(date +%T) lines $(wc -l < $O/seeds-$a.txt) | $(grep -acE 'keyed verify .*device draws' $O/seeds-$a.log) device-draw passes, $(grep -ac 'Stop {' $O/seeds-$a.log) stops, $(grep -ac 'Resume {' $O/seeds-$a.log) resumes | $(grep -aE 'requests in' $O/seeds-$a.log)"
done
for t in 1 0.6; do
  x=$(awk -v t=$t '$1==t' $O/seeds-0.txt | sort); y=$(awk -v t=$t '$1==t' $O/seeds-1.txt | sort)
  echo "seeds T $t: off $(echo "$x" | grep -c .) on $(echo "$y" | grep -c .) shared $(comm -12 <(echo "$x" | cut -d' ' -f1,2) <(echo "$y" | cut -d' ' -f1,2) | grep -c .) identical $(comm -12 <(echo "$x") <(echo "$y") | grep -c .)"
done
for sp in off def; do
  x=""; [ $sp = off ] && x="--spec off"
  perl -e 'alarm shift; exec @ARGV' 300 $BIN generate -m $M --backend metal --max-seq 8192 --profile-every 0 --show-ids --max-tokens 64 --prompt-ids $S/prompts/md5.json $x > $O/md5-$sp.out 2> $O/md5-$sp.err
  echo "greedy $sp rc $? ids-md5 $(grep -a '^ids\[' $O/md5-$sp.err | md5 -q) lines $(grep -ac '^ids\[' $O/md5-$sp.err) | $(grep -aE '^spec: verify' $O/md5-$sp.err | cut -c1-110)"
done
echo "end $(date +%T)"
