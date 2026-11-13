#!/usr/bin/env bash
# ab35 smoke: start the default arm on :8191 beside the production vLLM, probe it, read the spec counters, stop it.
set -uo pipefail
: "${FLEETQ_JOB:?}"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
rsh() { ssh -n -o BatchMode=yes spark2 "$@"; }
SPID=$(rsh "nohup ~/intarsia-ab35/bin/intarsia serve --backend cuda --model $M35 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 20480 --mem-budget 40 --max-batch 2 --profile-every 0 > ~/intarsia-ab35/out/serve-smoke.log 2>&1 < /dev/null & echo \$!")
WPID=$(rsh "nohup bash ~/intarsia-ab35/memwatch.sh $SPID smoke > /dev/null 2>&1 < /dev/null & echo \$!")
trap 'rsh "kill $WPID $SPID 2>/dev/null"' EXIT
T=$(date +%s)
for i in $(seq 1 150); do
  curl -sf http://spark-fa00:8191/health >/dev/null && break
  rsh "kill -0 $SPID" 2>/dev/null || { echo "server died"; rsh "tail -30 ~/intarsia-ab35/out/serve-smoke.log"; exit 1; }
  sleep 2
done
echo "up after $(( $(date +%s)-T ))s"
rsh "free -g | sed -n 2p; sort -k2 -n ~/intarsia-ab35/out/mem-smoke.txt | head -1"
( cd /opt/code/deepeval && uv run localeval probe --url http://spark-fa00:8191/v1 --model guidance --thinking off --no-unim )
curl -s http://spark-fa00:8191/v1/chat/completions -H 'content-type: application/json' -d '{"model":"guidance","messages":[{"role":"user","content":"Count from 1 to 40, separated by spaces."}],"max_tokens":120,"temperature":0}' | head -c 900; echo
curl -sf http://spark-fa00:8191/metrics | grep -E 'spec|batched' | head
rsh "grep -iE 'spec|mtp|max_batch|kv|chunk|macro' ~/intarsia-ab35/out/serve-smoke.log | head -20"
rsh "sort -k2 -n ~/intarsia-ab35/out/mem-smoke.txt | head -1"
