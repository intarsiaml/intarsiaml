#!/bin/zsh
# usage: leval6.sh <outdir> <fleetq label> <label1=bin1> <label2=bin2> ...
# One fleetq slot; each binary is served (`intarsia serve`, args of b4/leval4.sh) and measured with localeval speed
# back to back (same machine state). The GPU idle utilization is sampled before each server (waits above 10 percent).
OUT=${1:A}; FL=$2; shift 2  # absolute: localeval runs from /opt/code/deepeval
mkdir -p $OUT
JSON=$(fleetq acquire --host endor --exclusive --mem 24 --label "$FL" --expect 25 --wait --json)
echo "$JSON" > $OUT/fleetq.json
ID=$(echo "$JSON" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("id") or d.get("job",{}).get("id"))')
echo "fleetq job $ID"
SPID=""
cleanup() { [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 2; fleetq release $ID; }
trap cleanup EXIT
PORT=18311
util() { ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; }
for spec in "$@"; do
  LABEL=${spec%%=*}; BIN=${spec#*=}
  u=$(util); tries=0; while [ "${u:-99}" -gt 10 ] && [ $tries -lt 150 ]; do sleep 2; u=$(util); tries=$((tries+1)); done
  echo "$LABEL idle_util=$u(waits=$tries) zoom=$(pgrep -f zoom.us | wc -l | tr -d ' ')"
  FLEETQ_JOB=$ID $BIN serve --model /Users/josef/models/Qwen3-8B --backend metal --device 0 --host 127.0.0.1 --port $PORT --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui > $OUT/$LABEL-server.out 2> $OUT/$LABEL-server.err &
  SPID=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || { echo server died; exit 1; }; sleep 2; done
  echo "$LABEL server up"
  (cd /opt/code/deepeval && FLEETQ_JOB=$ID uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label $LABEL > $OUT/$LABEL-localeval.out 2> $OUT/$LABEL-localeval.err)
  echo "$LABEL localeval exit $?"
  kill $SPID; wait $SPID 2>/dev/null; SPID=""; sleep 3
done
