#!/bin/bash
# endor EXCLUSIVE fleetq slot P, lane ink-metal-dtopk: a per-dispatch GPU profile of speculative passes at the Metal default (DFlash2 k7 +
# cut 0.3), to rank the next lever. Diagnostic only (one command buffer per dispatch: times per op, not a speed figure).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$R/P; mkdir -p $O; T0=$(date +%s)
echo "slot P start $(date +%T) binary md5 $(md5 -q $B) commit $(cat $B.commit)"
for arm in dflt; do
  env INTARSIA_METAL_PROFILE=cb:$S/prof-$arm.jsonl INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_SPEC_TRACE=1 $(envs $arm) \
    $B generate -m $M --backend metal --profile-every 0 --show-ids --max-seq 20480 --max-tokens 128 --prompt-ids $R/prof4.json \
    $(flags $arm) > $O/p-$arm.out 2> $O/p-$arm.err
  echo "p-$arm rc $? t+$(( $(date +%s) - T0 ))s | trace $(wc -l < $S/prof-$arm.jsonl) records | $(grep -aE '^spec: verify' $O/p-$arm.err | head -1 | cut -c1-140)"
  python3 $R/prof.py $S/prof-$arm.jsonl $O/p-$arm.err 40 > $O/prof-$arm.md
  cat $O/prof-$arm.md
done
echo "slot P end $(date +%T) t+$(( $(date +%s) - T0 ))s"
