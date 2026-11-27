#!/bin/bash
# endor EXCLUSIVE fleetq slot C, lane ink-metal-dflash2-ready (serve's cold admission, ~4 min per plan, made `serve` too slow for slot A): check 2
# on `generate` (one process per arm and sampling config; 100 / 60 / 30 distinct 24-id prompts, one seed, 32 tokens) and check 4 (memory / KV
# capacity via generate's `--max-seq 262144 --kv-fraction 0.25`, default / dfl5 alternated).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$S/C; mkdir -p $O; T0=$(date +%s)
echo "slot C start $(date +%T) binary md5 $(md5 -q $B) git $(git -C $W rev-parse --short HEAD)"
macmon pipe -i 250 > $O/macmon.jsonl 2>/dev/null &
iso() { date -u +%Y-%m-%dT%H:%M:%S; }
echo "== check 2b: sampled, generate, seed 7, 32 tokens, --ignore-eos"
for arm in off dfl5 dfl5s; do
  a=${arm%s}; [ $arm = off ] && a=off; E=""
  [ $arm = dfl5 ] && E="INTARSIA_SPEC_TRACE=1"; [ $arm = dfl5s ] && E="INTARSIA_SPEC_TRACE=1 INTARSIA_SPEC_SAMPLED_DFLASH=1"
  for cfg in "1.0 0 1.0 100" "0.6 0 1.0 60" "0.7 20 0.9 30"; do
    set -- $cfg; T=$1; K=$2; P=$3; N=$4; f=$O/s-$arm-T$T-k$K
    env $E $B generate -m $M --backend metal --max-seq 4096 --profile-every 0 --show-ids --max-tokens 32 --ignore-eos \
      --temperature $T --top-k $K --top-p $P --seed 7 --prompt-ids $S/prompts/w$N.json $(flags $a) > $f.out 2> $f.err
    echo "$arm T$T k$K n$N rc $? t+$(( $(date +%s) - T0 ))s | ids lines $(grep -ac '^ids\[' $f.err) | verify/keyed trace lines $(grep -acE 'verify' $f.err)"
  done
done
grep -ah "sampled requests (temperature" $O/s-dfl5-T1.0-k0.err $O/s-dfl5s-T1.0-k0.err | sort | uniq -c
echo "== check 4: memory / KV capacity, generate --max-seq 4194304 (1048576 fallback) --kv-fraction 0.25, 64 tokens of the code prompt"
n=0
for arm in default dfl5 default dfl5; do
  n=$((n+1)); f=$O/mem$n-$arm; a=$(iso)
  for MS in ${MAXSEQ:-4194304 1048576}; do
    /usr/bin/time -l $B generate -m $M --backend metal --max-seq $MS --kv-fraction 0.25 --profile-every 0 --show-ids --max-tokens 64 --ignore-eos \
      --prompt-ids $S/prompts/c1.json $(flags $arm) > $f.out 2> $f.err
    rc=$?; [ $rc = 0 ] && break
  done
  echo "mem$n $arm rc $rc max-seq $MS window $a..$(iso) | $(grep -aE '^loaded ' $f.err | head -1 | cut -c1-260) | $(grep -aE 'peak memory footprint|maximum resident' $f.err | tr '\n' ' ')"
  sleep 3
done
echo "slot C end $(date +%T) t+$(( $(date +%s) - T0 ))s"
