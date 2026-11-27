#!/bin/bash
# endor EXCLUSIVE fleetq slot A, lane ink-metal-dflash2-ready: check 1 (greedy ids at 4k / 16k, three arms, `generate`) and check 2 (sampled
# requests on `serve`: arms off / default / dfl5 / dfl5s, seeds; plus the single-sequence solo greedy replies of the batch check).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$S/A; mkdir -p $O; T0=$(date +%s)
echo "slot A start $(date +%T) binary md5 $(md5 -q $B) git $(git -C $W rev-parse --short HEAD)"
macmon pipe -i 250 > $O/macmon.jsonl 2>/dev/null &
echo "== check 1: greedy, generate, code + len4k + len16k, 256 tokens, --ignore-eos"
for arm in off default dfl5; do
  /usr/bin/time -l $B generate -m $M --backend metal --max-seq 20480 --profile-every 0 --show-ids --max-tokens 256 --ignore-eos \
    --prompt-ids $S/prompts/g3.json $(flags $arm) > $O/g-$arm.out 2> $O/g-$arm.err
  echo "g-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode\[' $O/g-$arm.err | tr '\n' ' ' | cut -c1-300) | $(grep -aE 'peak memory footprint' $O/g-$arm.err)"
done
echo "== check 2: serve arms (sampled seeds, solo greedy b4)"
i=0
for spec in "off:off:" "default:default:" "dfl5:dfl5:INTARSIA_SPEC_TRACE=1" "dfl5s:dfl5:INTARSIA_SPEC_TRACE=1 INTARSIA_SPEC_SAMPLED_DFLASH=1"; do
  name=${spec%%:*}; rest=${spec#*:}; arm=${rest%%:*}; envs=${rest#*:}; i=$((i+1)); port=$((8180+i)); U=http://127.0.0.1:$port/v1
  [ $(( $(date +%s) - T0 )) -ge 780 ] && { echo "slot A: time box before arm $name"; break; }
  serve_start $name $arm $port "" "$envs"
  python3 $R/client.py wait $U 120 || { echo "arm $name: server not up"; serve_stop; continue; }
  echo "arm $name up t+$(( $(date +%s) - T0 ))s | $(grep -aE '^loaded ' $O/s-$name.err | head -1)"
  L0=$(wc -l < $O/s-$name.err)
  python3 $R/client.py seeds $U $O/seeds-$name.json 100 ${SEED_BUDGET:-150}
  L1=$(wc -l < $O/s-$name.err)
  echo "arm $name trace lines during the sampled phase: $((L1-L0)); keyed/verify lines: $(sed -n "$((L0+1)),${L1}p" $O/s-$name.err | grep -acE 'verify')"
  python3 $R/client.py greedy $U $O/solo-$name.json $S/prompts/b4.json 128
  serve_stop
  echo "arm $name done t+$(( $(date +%s) - T0 ))s | $(grep -aE 'peak memory footprint' $O/s-$name.err)"
  sleep 3
done
for a in default dfl5 dfl5s; do python3 $R/client.py cmp $O/seeds-off.json $O/seeds-$a.json; done
for a in default dfl5 dfl5s; do python3 $R/client.py cmp $O/solo-off.json $O/solo-$a.json; done
echo "slot A end $(date +%T) t+$(( $(date +%s) - T0 ))s"
