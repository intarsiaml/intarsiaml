#!/bin/bash
# endor EXCLUSIVE fleetq slot F, lane ink-metal-flip2: the pre-registered confirm gates F0-F4 of docs/results/2026-11-27-inkling2-metal-flip2.md
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$R/F; mkdir -p $O; T0=$(date +%s)
echo "slot F start $(date +%T) binary md5 $(md5 -q $B) commit $(cat $B.commit) git $(git -C $W rev-parse --short HEAD)"
macmon pipe -i 250 > $S/macmon-F.jsonl 2>/dev/null &
G() { env $(envs $1) /usr/bin/time -l $B generate -m $M --backend metal --profile-every 0 --show-ids "${@:2}" $(flags $1); }
echo "== F0: prologue split bitwise on the GPU (test exe)"
$S/bin/t-ps --test-threads 1 > $O/f0.txt 2>&1
echo "f0 rc $? | $(grep -aE '^test result' $O/f0.txt | tail -1) | $(grep -aE 'panicked' $O/f0.txt | head -2 | cut -c1-300)"
echo "== F1: greedy ids, g3.json (code, len4k, len16k), 256 tokens, --ignore-eos, prefill trace on"
for arm in off main flip; do
  env INTARSIA_INK_PREFILL_TRACE=1 $(envs $arm) /usr/bin/time -l $B generate -m $M --backend metal --profile-every 0 --show-ids --max-seq 20480 \
    --max-tokens 256 --ignore-eos --prompt-ids $R/g3.json $(flags $arm) > $O/g-$arm.out 2> $O/g-$arm.err
  echo "g-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify' $O/g-$arm.err | head -1 | cut -c1-160) | split traces $(grep -ac 'prologue split' $O/g-$arm.err) | $(grep -aE 'peak memory footprint' $O/g-$arm.err)"
  echo "g-$arm load: $(grep -aE 'DFlash2 drafter|^--spec dflash --spec-draft|^dflash: cut|^speculative decoding' $O/g-$arm.err | cut -c1-200 | tr '\n' '|')"
done
python3 $R/md5.py $O/g-off.err $O/g-main.err $O/g-flip.err
echo "== F2: sampled, seed 7, 32 tokens, --ignore-eos, T1.0 (w100) and T0.6 (w60): off vs flip"
for arm in off flip; do
  for cfg in "1.0 100" "0.6 60"; do
    set -- $cfg; f=$O/s-$arm-T$1
    /usr/bin/time -l $B generate -m $M --backend metal --max-seq 4096 --profile-every 0 --show-ids --max-tokens 32 --ignore-eos \
      --temperature $1 --top-k 0 --top-p 1.0 --seed 7 --prompt-ids $R/w$2.json $(flags $arm) > $f.out 2> $f.err
    echo "$arm T$1 n$2 rc $? t+$(( $(date +%s) - T0 ))s | ids lines $(grep -ac '^ids\[' $f.err) | $(grep -aE '^spec: verify' $f.err | head -1 | cut -c1-120)"
  done
done
python3 $R/seedcmp.py $O off flip
echo "== F3a: decode ABBA, speed.json (19 prompts, 256 tokens, EOS honoured), 45 s idle before each"
mkdir -p $O/d
for tok in main@r1 flip@r1 flip@r2 main@r2; do
  arm=${tok%@*}; f=$O/d/$tok
  sleep 45
  G $arm --max-seq 20480 --max-tokens 256 --prompt-ids $R/speed.json > $f.out 2> $f.err
  echo "$tok rc $? $(date +%T) t+$(( $(date +%s) - T0 ))s | $(grep -aoE 'dflash: cut [0-9.]+' $f.err | head -1) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err | head -1 | cut -c1-120)"
done
python3 $R/rule.py $O/d flip main
echo "== F3b: prefill ABBA, pf3.json (p1k, len4k, len16k), 1 token, 45 s idle before each (off = main, on = flip)"
mkdir -p $O/t; n=0
for arm in main flip flip main; do
  n=$((n+1)); a=$([ $arm = main ] && echo off || echo on); f=$O/t/r$(printf %02d $n)-$a
  sleep 45
  G $arm --max-seq 20480 --max-tokens 1 --prompt-ids $R/pf3.json > $f.out 2> $f.err
  echo "r$n $arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^prefill\[' $f.err | tr '\n' ' ' | cut -c1-300)"
done
python3 $R/pfabba.py $O/t
echo "== F4a: serve --max-batch 1, flip (no --spec) vs --spec off, multi-turn"
i=0
for arm in flip off; do
  i=$((i+1)); port=$((8440+i)); U=http://127.0.0.1:$port/v1; name=mb1-$arm
  t=$(date +%s)
  serve_start $name $arm $port "--max-batch 1"
  if python3 $R/client.py wait $U 420; then
    echo "$name up after $(( $(date +%s) - t )) s | $(grep -aE '^loaded ' $O/s-$name.err | head -1 | cut -c1-160) | $(grep -aE 'DFlash2 drafter|^--spec dflash --spec-draft|^dflash: cut|admission on plan' $O/s-$name.err | cut -c1-200 | tr '\n' '|')"
    python3 $R/client.py chat $U $O/chat-$name.json $R/b4.json $R/abba3.json
  else
    echo "$name NOT UP within 420 s"
  fi
  serve_stop; echo "$name stopped t+$(( $(date +%s) - T0 ))s"; sleep 3
done
python3 $R/client.py cmpchat $O/chat-mb1-flip.json $O/chat-mb1-off.json
echo "== F4b: serve --max-batch 4 (no --spec), solo + 2 x concurrent, vs #551's batch-4 replies"
port=8449; U=http://127.0.0.1:$port/v1; t=$(date +%s)
serve_start batch4 flip $port "--max-batch 4 --max-seq 4096"
if python3 $R/client.py wait $U 420; then
  echo "batch4 up after $(( $(date +%s) - t )) s | $(grep -aE '^loaded ' $O/s-batch4.err | head -1 | cut -c1-160)"
  python3 $R/client.py greedy $U $O/b4-solo.json $R/b4.json 128
  python3 $R/client.py greedy $U $O/b4-conc.json $R/b4.json 128 conc
  python3 $R/client.py greedy $U $O/b4-conc2.json $R/b4.json 128 conc
  python3 $R/client.py cmp $O/b4-solo.json $O/b4-conc.json; python3 $R/client.py cmp $O/b4-solo.json $O/b4-conc2.json
  python3 $R/client.py cmp $W/docs/results/2026-11-27-inkling2-metal-dflash2-flip/raw/G/b4-solo.json $O/b4-solo.json
else
  echo "batch4 NOT UP within 420 s"
fi
serve_stop
echo "slot F end $(date +%T) t+$(( $(date +%s) - T0 ))s"
