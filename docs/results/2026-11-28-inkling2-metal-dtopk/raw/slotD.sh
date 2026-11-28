#!/bin/bash
# endor EXCLUSIVE fleetq slot D, lane ink-metal-dtopk: the pre-registered gates D0-D4 of docs/results/2026-11-28-inkling2-metal-dtopk.md
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$R/D; mkdir -p $O/d; T0=$(date +%s)
echo "slot D start $(date +%T) binary md5 $(md5 -q $B) commit $(cat $B.commit) git $(git -C $W rev-parse --short HEAD)"
macmon pipe -i 250 > $S/macmon-D.jsonl 2>/dev/null &
G() { env $(envs $1) /usr/bin/time -l $B generate -m $M --backend metal --profile-every 0 --show-ids "${@:2}" $(flags $1); }
echo "== D0: draft phase trace, speed.json, 256 tokens, EOS honoured, INTARSIA_SPEC_TRACE=1 (no conf trace): dflt vs dtopk"
for arm in dflt dtopk; do
  env INTARSIA_SPEC_TRACE=1 $(envs $arm) $B generate -m $M --backend metal --profile-every 0 --show-ids --max-seq 20480 --max-tokens 256 \
    --prompt-ids $R/speed.json > $O/t-$arm.out 2> $O/t-$arm.err
  echo "t-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE 'dflash: candidates|^dflash: cut' $O/t-$arm.err | cut -c1-120 | tr '\n' '|')"
done
python3 $R/phases.py $O/t-dflt.err $O/t-dtopk.err
echo "== D1: candidates check, g3.json + speed.json, INTARSIA_DFLASH_DEVICE_TOPK=check"
for set in g3 speed; do
  extra=$([ $set = g3 ] && echo "--max-tokens 256 --ignore-eos" || echo "--max-tokens 256")
  G chk --max-seq 20480 $extra --prompt-ids $R/$set.json > $O/c-$set.out 2> $O/c-$set.err
  echo "c-$set rc $? t+$(( $(date +%s) - T0 ))s | checks $(grep -ac 'dflash2 topk check: rank 0 rows' $O/c-$set.err) | rows differing $(grep -aoE 'rows [0-9]+ differing [0-9]+' $O/c-$set.err | awk '{s+=$4} END {print s+0}') | $(grep -aE '^spec: verify' $O/c-$set.err | head -1 | cut -c1-120)"
done
echo "== D2: greedy ids, g3.json (code, len4k, len16k), 256 tokens, --ignore-eos"
for arm in off dflt dtopk; do
  G $arm --max-seq 20480 --max-tokens 256 --ignore-eos --prompt-ids $R/g3.json > $O/g-$arm.out 2> $O/g-$arm.err
  echo "g-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify' $O/g-$arm.err | head -1 | cut -c1-140) | $(grep -aE 'peak memory footprint' $O/g-$arm.err)"
done
python3 $R/md5.py $O/g-off.err $O/g-dflt.err $O/g-dtopk.err
echo "== D3: decode ABBA x 4, speed.json (19 prompts, 256 tokens, EOS honoured), 45 s idle before each"
for tok in dflt@r1 dtopk@r1 dtopk@r2 dflt@r2 dflt@r3 dtopk@r3 dtopk@r4 dflt@r4; do
  arm=${tok%@*}; f=$O/d/$tok
  sleep 45
  G $arm --max-seq 20480 --max-tokens 256 --prompt-ids $R/speed.json > $f.out 2> $f.err
  echo "$tok rc $? $(date +%T) t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err | head -1 | cut -c1-120)"
done
RULE_BAR=1.01 python3 $R/rule.py $O/d dtopk dflt
echo "== D4: serve --max-batch 1, dtopk vs --spec off, multi-turn"
i=0
for arm in dtopk off; do
  i=$((i+1)); port=$((8450+i)); U=http://127.0.0.1:$port/v1; name=mb1-$arm
  t=$(date +%s)
  serve_start $name $arm $port "--max-batch 1"
  if python3 $R/client.py wait $U 420; then
    echo "$name up after $(( $(date +%s) - t )) s | $(grep -aE 'dflash: candidates|^dflash: cut|admission on plan' $O/s-$name.err | cut -c1-160 | tr '\n' '|')"
    python3 $R/client.py chat $U $O/chat-$name.json $R/b4.json $R/abba3.json
  else
    echo "$name NOT UP within 420 s"
  fi
  serve_stop; echo "$name stopped t+$(( $(date +%s) - T0 ))s"; sleep 3
done
python3 $R/client.py cmpchat $O/chat-mb1-dtopk.json $O/chat-mb1-off.json
echo "slot D end $(date +%T) t+$(( $(date +%s) - T0 ))s"
