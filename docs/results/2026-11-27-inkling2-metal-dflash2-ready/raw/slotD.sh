#!/bin/bash
# endor EXCLUSIVE fleetq slot D, lane ink-metal-dflash2-ready: check 3a (--max-batch 4 --spec dflash refusal), check 2c (ABBA, information),
# check 3b (serve --max-batch 4, no --spec: 4 concurrent greedy vs solo; time-boxed, serve's admission is cold for a new plan, ~4 min).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$S/D; mkdir -p $O; T0=$(date +%s)
echo "slot D start $(date +%T) binary md5 $(md5 -q $B) git $(git -C $W rev-parse --short HEAD)"
macmon pipe -i 250 > $O/macmon.jsonl 2>/dev/null &
echo "== 3a: serve --max-batch 4 --spec dflash --spec-draft D --spec-k 5"
t=$(date +%s); $B serve -m $M --backend metal --no-supervisor --no-ui --port 8191 --max-batch 4 --spec dflash --spec-draft $D --spec-k 5 > $O/refuse.out 2> $O/refuse.err; rc=$?
echo "3a rc $rc after $(( $(date +%s) - t )) s | $(tail -3 $O/refuse.err | tr '\n' '|' | cut -c1-500)"
echo "== 2c: ABBA speed, abba3.json (warm + 3 short), 256 tokens, EOS honoured, 45 s idle"
for tok in dspark@r1 dfl5@r1 dfl5@r2 dspark@r2; do
  arm=${tok%@*}; f=$O/abba-$tok; F=""; [ $arm = dfl5 ] && F=$(flags dfl5)
  sleep 45
  python3 -c 'import time; print(f"{time.time():.3f}")' > $f.t
  $B generate -m $M --backend metal --max-seq 20480 --profile-every 0 --show-ids --max-tokens 256 --prompt-ids $S/prompts/abba3.json $F > $f.out 2> $f.err
  rc=$?; python3 -c 'import time; print(f"{time.time():.3f}")' >> $f.t
  echo "$tok rc $rc t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode\[' $f.err | tr '\n' ' ' | cut -c1-400)"
done
echo "== 3b: serve --max-batch 4 (no --spec), time box ${WAIT_B:-330} s for the server to come up"
U=http://127.0.0.1:8192/v1
serve_start batch4 default 8192 "--max-batch 4 --max-seq 4096" ""
t=$(date +%s)
if python3 $R/client.py wait $U ${WAIT_B:-330}; then
  echo "batch4 up after $(( $(date +%s) - t )) s | $(grep -aE '^loaded ' $O/s-batch4.err | head -1) | $(grep -aiE 'spec|draft|batching' $O/s-batch4.err | head -8 | tr '\n' '|' | cut -c1-800)"
  python3 $R/client.py greedy $U $O/b4-solo.json $S/prompts/b4.json 128
  python3 $R/client.py greedy $U $O/b4-conc.json $S/prompts/b4.json 128 conc
  python3 $R/client.py greedy $U $O/b4-conc2.json $S/prompts/b4.json 128 conc
  python3 $R/client.py cmp $O/b4-solo.json $O/b4-conc.json; python3 $R/client.py cmp $O/b4-solo.json $O/b4-conc2.json
else
  echo "batch4: NOT UP within the time box (cold admission)"
fi
serve_stop
echo "slot D end $(date +%T) t+$(( $(date +%s) - T0 ))s"
