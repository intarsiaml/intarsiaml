#!/bin/bash
# endor EXCLUSIVE fleetq slot G, lane ink-metal-dgemv: the pre-registered checks G0-G3 of docs/results/2026-11-28-inkling2-metal-dgemv.md
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$R/G; mkdir -p $O/d; T0=$(date +%s)
echo "slot G start $(date +%T) binary md5 $(md5 -q $B) commit $(cat $B.commit)"
macmon pipe -i 250 > $S/macmon-G.jsonl 2>/dev/null &
G() { env $(envs $1) /usr/bin/time -l $B generate -m $M --backend metal --profile-every 0 --show-ids "${@:2}" $(flags $1); }
echo "== G0: per-dispatch cb profile, prof4.json, 128 tokens: rk0 (dflt's is slot P of lane ink-metal-dtopk, same binary)"
for arm in rk0; do
  env INTARSIA_METAL_PROFILE=cb:$S/prof-$arm.jsonl INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_SPEC_TRACE=1 $(envs $arm) \
    $B generate -m $M --backend metal --profile-every 0 --show-ids --max-seq 20480 --max-tokens 128 --prompt-ids $R/prof4.json \
    $(flags $arm) > $O/p-$arm.out 2> $O/p-$arm.err
  echo "p-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify' $O/p-$arm.err | head -1 | cut -c1-140)"
  python3 $R/prof.py $S/prof-$arm.jsonl $O/p-$arm.err 45 > $O/prof-$arm.md; head -30 $O/prof-$arm.md
done
echo "== G1: draft phase trace, speed.json, 256 tokens, EOS honoured, INTARSIA_SPEC_TRACE=1: dflt vs rk0"
for arm in dflt rk0; do
  env INTARSIA_SPEC_TRACE=1 $(envs $arm) $B generate -m $M --backend metal --profile-every 0 --show-ids --max-seq 20480 --max-tokens 256 \
    --prompt-ids $R/speed.json > $O/t-$arm.out 2> $O/t-$arm.err
  echo "t-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify' $O/t-$arm.err | head -1 | cut -c1-140)"
done
python3 $R/phases.py $O/t-dflt.err $O/t-rk0.err
echo "== G2: greedy ids, g3.json (code, len4k, len16k), 256 tokens, --ignore-eos"
for arm in off dflt rk0; do
  G $arm --max-seq 20480 --max-tokens 256 --ignore-eos --prompt-ids $R/g3.json > $O/g-$arm.out 2> $O/g-$arm.err
  echo "g-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify' $O/g-$arm.err | head -1 | cut -c1-140)"
done
python3 $R/md5.py $O/g-off.err $O/g-dflt.err $O/g-rk0.err
echo "== G3: decode ABBA x 4, speed.json (19 prompts, 256 tokens, EOS honoured), 45 s idle before each"
for tok in dflt@r1 rk0@r1 rk0@r2 dflt@r2 dflt@r3 rk0@r3 rk0@r4 dflt@r4; do
  arm=${tok%@*}; f=$O/d/$tok
  sleep 45
  G $arm --max-seq 20480 --max-tokens 256 --prompt-ids $R/speed.json > $f.out 2> $f.err
  echo "$tok rc 0 $(date +%T) t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err | head -1 | cut -c1-120)"
done
RULE_BAR=1.01 python3 $R/rule.py $O/d rk0 dflt
echo "slot G end $(date +%T) t+$(( $(date +%s) - T0 ))s"
