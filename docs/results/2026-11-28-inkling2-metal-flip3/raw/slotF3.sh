#!/bin/bash
# endor EXCLUSIVE fleetq slot F3, lane ink-metal-flip3: the pre-registered checks F1-F3 of docs/results/2026-11-28-inkling2-metal-flip3.md
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$R/F3; mkdir -p $O/d; T0=$(date +%s)
echo "slot F3 start $(date +%T) binary md5 $(md5 -q $B) commit $(cat $B.commit)"
macmon pipe -i 250 > $S/macmon-H.jsonl 2>/dev/null &
G() { env $(envs $1) /usr/bin/time -l $B generate -m $M --backend metal --profile-every 0 --show-ids "${@:2}" $(flags $1); }
echo "== F1: draft phase trace + the default notes, speed.json, 256 tokens, INTARSIA_SPEC_TRACE=1: rev vs new"
for arm in rev new; do
  env INTARSIA_SPEC_TRACE=1 $(envs $arm) $B generate -m $M --backend metal --profile-every 0 --show-ids --max-seq 20480 --max-tokens 256 \
    --prompt-ids $R/speed.json > $O/t-$arm.out 2> $O/t-$arm.err
  echo "t-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify' $O/t-$arm.err | head -1 | cut -c1-140)"
  grep -aE '^dflash: DFlash2 draft head|^dflash: candidates by the Metal device' $O/t-$arm.err | head -3
done
python3 $R/phases.py $O/t-rev.err $O/t-new.err
python3 $R/md5.py $O/t-rev.err $O/t-new.err
echo "== F2: greedy ids, g3.json (code, len4k, len16k), 256 tokens, --ignore-eos"
for arm in off rev new; do
  G $arm --max-seq 20480 --max-tokens 256 --ignore-eos --prompt-ids $R/g3.json > $O/g-$arm.out 2> $O/g-$arm.err
  echo "g-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify' $O/g-$arm.err | head -1 | cut -c1-140)"
done
python3 $R/md5.py $O/g-off.err $O/g-rev.err $O/g-new.err
echo "== F3: decode ABBA x 4, speed.json (19 prompts, 256 tokens, EOS honoured), 45 s idle before each"
for tok in rev@r1 new@r1 new@r2 rev@r2 rev@r3 new@r3 new@r4 rev@r4; do
  arm=${tok%@*}; f=$O/d/$tok
  sleep 45
  G $arm --max-seq 20480 --max-tokens 256 --prompt-ids $R/speed.json > $f.out 2> $f.err
  echo "$tok rc 0 $(date +%T) t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err | head -1 | cut -c1-120)"
done
RULE_BAR=1.01 python3 $R/rule.py $O/d new rev
echo "slot F3 end $(date +%T) t+$(( $(date +%s) - T0 ))s"
