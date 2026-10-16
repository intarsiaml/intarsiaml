#!/bin/zsh
# localeval speed, one server at a time, base new new base, inside the caller's fleetq slot (FLEETQ_JOB set).
# base = INTARSIA_NAX_PICK=p45 (tile pick before the pad-aware rule), new = default pick, same binary.
#   localeval-abba.sh <outdir-name> <prefill chunk> <sizes> [reps]
W=/opt/code/intarsiaml-gemm-midm
OUT=$W/docs/results/raw/2026-10-15-metal-gemm-midm/$1
CHUNK=$2
SIZES=$3
REPS=${4:-3}
BIN=${INTARSIA_BIN:-$W/target/release/intarsia}
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
source $W/docs/results/raw/2026-10-15-metal-gemm-midm/waitidle.sh
mkdir -p $OUT
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
pgrep -fl zoom.us > $OUT/zoom.txt 2>&1
PORT=8260
for B in base new new base; do
  PORT=$((PORT+1))
  N=$(ls $OUT | grep -c "^gm-c$CHUNK-$B-.*-server.err")
  L=gm-c$CHUNK-$B-$((N+1))
  if [ $B = base ]; then export INTARSIA_NAX_PICK=p45; else unset INTARSIA_NAX_PICK; fi
  $BIN serve --model $M --backend metal --device 0 --host 127.0.0.1 --port $PORT --served-model-name intarsia-metal --max-seq 8192 \
    --prefill-chunk $CHUNK --profile-every 0 --no-supervisor --no-ui > $OUT/$L-server.out 2> $OUT/$L-server.err &
  SPID=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || { echo "server died"; break; }; sleep 2; done
  sleep 3
  wait_idle || touch $OUT/$L.contaminated
  echo "idle before $L: $(idle3)" | tee -a $OUT/idle.txt
  macmon pipe -i 200 > $OUT/$L.macmon.jsonl 2>/dev/null &
  MP=$!
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia-metal --sizes $SIZES --gen 128 --reps $REPS \
    --thinking off --label $L) > $OUT/$L-localeval.out 2> $OUT/$L-localeval.err
  echo "$L localeval exit $?"
  kill $MP 2>/dev/null
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 3
done
echo done
