#!/bin/bash
# endor EXCLUSIVE fleetq slot H, lane ink-metal-p12head: the pre-registered checks H1-H4 of docs/results/2026-11-28-inkling2-metal-p12head.md
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$R/H; mkdir -p $O/d; T0=$(date +%s)
echo "slot H start $(date +%T) binary md5 $(md5 -q $B) commit $(cat $B.commit)"
macmon pipe -i 250 > $S/macmon-H.jsonl 2>/dev/null &
G() { env $(envs $1) /usr/bin/time -l $B generate -m $M --backend metal --profile-every 0 --show-ids "${@:2}" $(flags $1); }
echo "== H1 + H3: draft phase trace, speed.json, 256 tokens, EOS honoured, INTARSIA_SPEC_TRACE=1: dflt vs p12"
for arm in dflt p12; do
  env INTARSIA_SPEC_TRACE=1 $(envs $arm) $B generate -m $M --backend metal --profile-every 0 --show-ids --max-seq 20480 --max-tokens 256 \
    --prompt-ids $R/speed.json > $O/t-$arm.out 2> $O/t-$arm.err
  echo "t-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify' $O/t-$arm.err | head -1 | cut -c1-140)"
  grep -aE '^dflash: DFlash2 draft head' $O/t-$arm.err | head -2
done
python3 $R/phases.py $O/t-dflt.err $O/t-p12.err
python3 $R/md5.py $O/t-dflt.err $O/t-p12.err
echo "== H2: greedy ids, g3.json (code, len4k, len16k), 256 tokens, --ignore-eos"
for arm in off dflt p12; do
  G $arm --max-seq 20480 --max-tokens 256 --ignore-eos --prompt-ids $R/g3.json > $O/g-$arm.out 2> $O/g-$arm.err
  echo "g-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify' $O/g-$arm.err | head -1 | cut -c1-140)"
done
python3 $R/md5.py $O/g-off.err $O/g-dflt.err $O/g-p12.err
echo "== H4: decode ABBA x 4, speed.json (19 prompts, 256 tokens, EOS honoured), 45 s idle before each"
for tok in dflt@r1 p12@r1 p12@r2 dflt@r2 dflt@r3 p12@r3 p12@r4 dflt@r4; do
  arm=${tok%@*}; f=$O/d/$tok
  sleep 45
  G $arm --max-seq 20480 --max-tokens 256 --prompt-ids $R/speed.json > $f.out 2> $f.err
  echo "$tok rc 0 $(date +%T) t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err | head -1 | cut -c1-120)"
done
RULE_BAR=1.01 python3 $R/rule.py $O/d p12 dflt
echo "slot H end $(date +%T) t+$(( $(date +%s) - T0 ))s"
