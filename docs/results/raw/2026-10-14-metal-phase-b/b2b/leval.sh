#!/bin/zsh
# usage: leval.sh <binary> <label> <port> <outdir>
BIN=$1; LABEL=$2; PORT=$3; OUT=$4
mkdir -p $OUT
JSON=$(fleetq acquire --host endor --exclusive --mem 24 --label "metal B2 fusion localeval $LABEL" --expect 20 --wait --json)
echo "$JSON" > $OUT/$LABEL.fleetq.json
ID=$(echo "$JSON" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("id") or d.get("job",{}).get("id"))')
echo "fleetq job $ID"
SPID=""
cleanup() { [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 2; fleetq release $ID; }
trap cleanup EXIT
FLEETQ_JOB=$ID $BIN serve --model /Users/josef/models/Qwen3-8B --backend metal --device 0 --host 127.0.0.1 --port $PORT --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui > $OUT/$LABEL-server.out 2> $OUT/$LABEL-server.err &
SPID=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || { echo server died; exit 1; }; sleep 2; done
echo server up
cd /opt/code/deepeval && FLEETQ_JOB=$ID uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label $LABEL > $OUT/$LABEL-localeval.out 2> $OUT/$LABEL-localeval.err
echo localeval exit $?
