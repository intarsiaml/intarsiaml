#!/bin/bash
# endor EXCLUSIVE fleetq slot B, lane ink-metal-dflash2-ready: check 3 (batching: refusal of --spec dflash --max-batch 4; plain batched serve,
# 4 concurrent greedy vs solo), check 4 (memory and KV capacity, default / dfl5 alternated), check 2c (ABBA on 3 short prompts, information).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$S/B; mkdir -p $O; T0=$(date +%s)
echo "slot B start $(date +%T) binary md5 $(md5 -q $B) git $(git -C $W rev-parse --short HEAD)"
macmon pipe -i 250 > $O/macmon.jsonl 2>/dev/null &
iso() { date -u +%Y-%m-%dT%H:%M:%S; }
memsnap() { macmon pipe -s 2 -i 250 2>/dev/null | tail -1 | python3 -c 'import sys,json; d=json.loads(sys.stdin.readline()); print(round(d["memory"]["ram_usage"]/2**30,2))'; }
echo "== 3a: --max-batch 4 --spec dflash --spec-draft D --spec-k 5"
t=$(date +%s); $B serve -m $M --backend metal --no-supervisor --no-ui --port 8191 --max-batch 4 --spec dflash --spec-draft $D --spec-k 5 > $O/refuse.out 2> $O/refuse.err; rc=$?
echo "3a rc $rc after $(( $(date +%s) - t )) s | $(tail -3 $O/refuse.err | tr '\n' '|' | cut -c1-400)"
sleep 3
echo "== 3b: --max-batch 4, no --spec: 4 concurrent greedy vs solo"
U=http://127.0.0.1:8192/v1
serve_start batch4 default 8192 "--max-batch 4" ""
python3 $R/client.py wait $U 120 && {
  echo "batch4 up | $(grep -aE '^loaded ' $O/s-batch4.err | head -1) | $(grep -aiE 'spec|draft|batch' $O/s-batch4.err | head -8 | tr '\n' '|' | cut -c1-700)"
  python3 $R/client.py greedy $U $O/b4-solo.json $S/prompts/b4.json 128
  python3 $R/client.py greedy $U $O/b4-conc.json $S/prompts/b4.json 128 conc
  python3 $R/client.py greedy $U $O/b4-conc2.json $S/prompts/b4.json 128 conc
}
serve_stop
echo "3b done t+$(( $(date +%s) - T0 ))s"
sleep 3
echo "== 4: memory / KV capacity, serve --max-seq 262144 --kv-fraction 0.25, default dfl5 default dfl5"
n=0
for arm in default dfl5 default dfl5; do
  n=$((n+1)); port=$((8200+n)); U=http://127.0.0.1:$port/v1
  m0=$(memsnap); a=$(iso)
  serve_start mem$n-$arm $arm $port "--max-seq 262144 --kv-fraction 0.25" ""
  python3 $R/client.py wait $U 180 || { echo "mem$n $arm: not up"; serve_stop; continue; }
  m1=$(memsnap)
  python3 $R/client.py greedy $U $O/mem$n-$arm.json $S/prompts/b4.json 64 > /dev/null
  m2=$(memsnap); b=$(iso)
  echo "mem$n $arm: ram_usage GiB before $m0, loaded $m1, after requests $m2 | window $a..$b | $(grep -aE '^loaded ' $O/s-mem$n-$arm.err | head -1)"
  serve_stop
  echo "mem$n $arm exit | $(grep -aE 'peak memory footprint|maximum resident' $O/s-mem$n-$arm.err | tr '\n' ' ')"
  sleep 3
done
echo "== 2c: ABBA speed, abba3.json (warm + 3 short), 256 tokens, EOS honoured, 45 s idle"
for tok in dspark@r1 dfl5@r1 dfl5@r2 dspark@r2; do
  arm=${tok%@*}; f=$O/abba-$tok; F=""; [ $arm = dfl5 ] && F=$(flags dfl5)
  sleep 45
  python3 -c 'import time; print(f"{time.time():.3f}")' > $f.t
  $B generate -m $M --backend metal --max-seq 20480 --profile-every 0 --show-ids --max-tokens 256 --prompt-ids $S/prompts/abba3.json $F > $f.out 2> $f.err
  rc=$?; python3 -c 'import time; print(f"{time.time():.3f}")' >> $f.t
  echo "$tok rc $rc t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode\[' $f.err | tr '\n' ' ' | cut -c1-400)"
done
echo "slot B end $(date +%T) t+$(( $(date +%s) - T0 ))s"
