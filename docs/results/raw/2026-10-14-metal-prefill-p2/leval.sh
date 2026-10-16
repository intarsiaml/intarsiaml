#!/bin/zsh
# P2 same-slot localeval: pre, new, then the reversed pair new, pre (one fleetq slot, one server per run).
# usage: leval.sh <outdir> <pre-binary> <new-binary> [pair-order ...]   default order: "pre new new pre"
OUT=$1; PRE=$2; NEW=$3; shift 3
ORDER=(${@:-pre new new pre})
[ $# -eq 0 ] && ORDER=(pre new new pre)
mkdir -p $OUT
JSON=$(fleetq acquire --host endor --exclusive --mem 24 --label "metal P2 flash prefill localeval ${ORDER}" --expect 40 --wait --json)
echo "$JSON" > $OUT/fleetq.json
ID=$(echo "$JSON" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("id") or d.get("job",{}).get("id"))')
echo "fleetq job $ID"
SPID=""
cleanup() { [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 2; fleetq release $ID; }
trap cleanup EXIT
for i in 1 2 3; do ioreg -r -d 1 -w0 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tee $OUT/gpu-idle-before.txt
PORT=8650
n=0
for L in $ORDER; do
  n=$((n+1))
  BIN=$NEW; [ $L = pre ] && BIN=$PRE
  TAG=run$n-$L
  PORT=$((PORT+1))
  FLEETQ_JOB=$ID $BIN serve --model /Users/josef/models/Qwen3-8B --backend metal --device 0 --host 127.0.0.1 --port $PORT --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui > $OUT/$TAG-server.out 2> $OUT/$TAG-server.err &
  SPID=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || { echo server died; exit 1; }; sleep 2; done
  echo "server up: $TAG"
  (cd /opt/code/deepeval && FLEETQ_JOB=$ID uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label p2-$TAG > $OUT/$TAG-localeval.out 2> $OUT/$TAG-localeval.err)
  echo "localeval $TAG exit $?"
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; SPID=""
  sleep 3
done
echo done
