#!/bin/bash
# endor EXCLUSIVE fleetq slot G, lane ink-metal-flip: the pre-registered gates G1-G6 of docs/results/2026-11-27-inkling2-metal-dflash2-flip.md
# on one binary (the branch; arm main = INTARSIA_SPEC_DFLASH2_DEFAULT=0 is today's default).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$R/G; mkdir -p $O; T0=$(date +%s)
echo "slot G start $(date +%T) binary md5 $(md5 -q $B) commit $(cat $B.commit) git $(git -C $W rev-parse --short HEAD)"
macmon pipe -i 250 > $S/macmon-G.jsonl 2>/dev/null &
G() { env $(envs $1) /usr/bin/time -l $B generate -m $M --backend metal --profile-every 0 --show-ids "${@:2}" $(flags $1); }
echo "== G1 + G2: greedy, generate, code + len4k + len16k, 256 tokens, --ignore-eos"
for arm in flip off main dfl5; do
  G $arm --max-seq 20480 --max-tokens 256 --ignore-eos --prompt-ids $R/g3.json > $O/g-$arm.out 2> $O/g-$arm.err
  echo "g-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode\[' $O/g-$arm.err | tr '\n' ' ' | cut -c1-300) | $(grep -aE '^spec: verify' $O/g-$arm.err | head -1 | cut -c1-160) | $(grep -aE 'peak memory footprint' $O/g-$arm.err)"
  echo "g-$arm load: $(grep -aE 'DFlash2 drafter|second draft|DSpark default|^--spec dflash|^dflash: ' $O/g-$arm.err | tr '\n' '|' | cut -c1-700)"
done
python3 $R/md5.py $O/g-off.err $O/g-main.err $O/g-flip.err $O/g-dfl5.err
echo "== G3: sampled, generate, seed 7, 32 tokens, --ignore-eos (off vs flip; flip traced)"
for arm in off flip; do
  E=""; [ $arm = flip ] && E="INTARSIA_SPEC_TRACE=1"
  for cfg in "1.0 0 1.0 100" "0.6 0 1.0 60" "0.7 20 0.9 30"; do
    set -- $cfg; f=$O/s-$arm-T$1-k$2
    env $E /usr/bin/time -l $B generate -m $M --backend metal --max-seq 4096 --profile-every 0 --show-ids --max-tokens 32 --ignore-eos \
      --temperature $1 --top-k $2 --top-p $3 --seed 7 --prompt-ids $R/w$4.json $(flags $arm) > $f.out 2> $f.err
    echo "$arm T$1 k$2 n$4 rc $? t+$(( $(date +%s) - T0 ))s | ids lines $(grep -ac '^ids\[' $f.err) | verify trace lines $(grep -acE 'verify' $f.err)"
  done
done
python3 $R/seedcmp.py $O off flip
echo "== G5: memory / KV capacity, generate --max-seq 4194304 (1048576 fallback) --kv-fraction 0.25, 64 tokens of the code prompt"
n=0
for arm in main flip dfl5 main flip dfl5; do
  n=$((n+1)); f=$O/mem$n-$arm
  for MS in 4194304 1048576; do
    G $arm --max-seq $MS --kv-fraction 0.25 --max-tokens 64 --ignore-eos --prompt-ids $R/c1.json > $f.out 2> $f.err
    rc=$?; [ $rc = 0 ] && break
  done
  echo "mem$n $arm rc $rc max-seq $MS | $(grep -aE '^loaded ' $f.err | head -1 | cut -c1-260) | $(grep -aE 'peak memory footprint|maximum resident' $f.err | tr '\n' ' ')"
  sleep 3
done
echo "== G6: ABBA speed, abba3.json (warm + 3 short), 256 tokens, EOS honoured, 45 s idle"
for tok in main@r1 flip@r1 flip@r2 main@r2; do
  arm=${tok%@*}; f=$O/abba-$tok
  sleep 45
  G $arm --max-seq 20480 --max-tokens 256 --prompt-ids $R/abba3.json > $f.out 2> $f.err
  echo "$tok rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode\[' $f.err | tr '\n' ' ' | cut -c1-400) | $(grep -aE '^spec: verify' $f.err | head -1 | cut -c1-120)"
done
mkdir -p $O/abba; mv $O/abba-*.err $O/abba/; python3 $R/abba.py $O/abba
echo "== G4: serve --max-batch 4 (no --spec), time box ${WAIT_B:-420} s for the server to come up"
U=http://127.0.0.1:8392/v1
/usr/bin/time -l $B serve -m $M --backend metal --no-supervisor --no-ui --port 8392 --max-batch 4 --max-seq 4096 > $O/s-batch4.out 2> $O/s-batch4.err &
SPID=$!; t=$(date +%s)
if python3 $R/client.py wait $U ${WAIT_B:-420}; then
  echo "batch4 up after $(( $(date +%s) - t )) s | $(grep -aE '^loaded ' $O/s-batch4.err | head -1) | $(grep -aiE 'spec|draft|DFlash|DSpark' $O/s-batch4.err | head -8 | tr '\n' '|' | cut -c1-800)"
  python3 $R/client.py greedy $U $O/b4-solo.json $R/b4.json 128
  python3 $R/client.py greedy $U $O/b4-conc.json $R/b4.json 128 conc
  python3 $R/client.py greedy $U $O/b4-conc2.json $R/b4.json 128 conc
  python3 $R/client.py cmp $O/b4-solo.json $O/b4-conc.json; python3 $R/client.py cmp $O/b4-solo.json $O/b4-conc2.json
else
  echo "batch4: NOT UP within the time box"
fi
for c in $(pgrep -P $SPID 2>/dev/null); do kill -TERM $c; done; sleep 5; wait $SPID 2>/dev/null
echo "slot G end $(date +%T) t+$(( $(date +%s) - T0 ))s"
