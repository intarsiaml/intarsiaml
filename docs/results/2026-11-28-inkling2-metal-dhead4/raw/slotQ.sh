#!/bin/bash
# endor EXCLUSIVE fleetq slot Q, lane ink-metal-dhead4: the probe. Does a quantised draft head copy (mxfp4 / fp8) take an M-row kernel on
# Metal, and does the draft head phase drop clearly below the P12 head's (1.476 ms a pass, slot H)? One traced pass set per arm + g3 ids.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$R/Q; mkdir -p $O; T0=$(date +%s)
echo "slot Q start $(date +%T) binary md5 $(md5 -q $B) commit $(cat $B.commit)"
for arm in both mx f8; do
  env INTARSIA_SPEC_TRACE=1 $(envs $arm) $B generate -m $M --backend metal --profile-every 0 --show-ids --max-seq 20480 --max-tokens 256 \
    --prompt-ids $R/speed.json > $O/t-$arm.out 2> $O/t-$arm.err
  echo "t-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify' $O/t-$arm.err | head -1 | cut -c1-140) | $(grep -aE '^decode overall' $O/t-$arm.err)"
  grep -aE '^dflash: DFlash2 draft head|^Error|^error' $O/t-$arm.err | head -3
done
python3 $R/phases.py $O/t-both.err $O/t-mx.err $O/t-f8.err
python3 $R/md5.py $O/t-both.err $O/t-mx.err $O/t-f8.err
for arm in mx f8; do
  env $(envs $arm) $B generate -m $M --backend metal --profile-every 0 --show-ids --max-seq 20480 --max-tokens 256 --ignore-eos --prompt-ids $R/g3.json > $O/g-$arm.out 2> $O/g-$arm.err
  echo "g-$arm rc $? | $(grep -aE '^spec: verify' $O/g-$arm.err | head -1 | cut -c1-140)"
done
python3 $R/md5.py $O/g-mx.err $O/g-f8.err
echo "goldens: code c69d1438ab3e715feb93a0614ebddcfe len4k 9e85ba1117e4a6a6cc12c5c917532e7f len16k 89994b876223335763b2004d0d281157"
echo "slot Q end $(date +%T) t+$(( $(date +%s) - T0 ))s"
