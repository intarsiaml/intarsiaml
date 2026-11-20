#!/usr/bin/env bash
# kvonce nsys window: kernel-level attribution of single-stream decode. One server under `nsys launch` (CUDA trace, graph nodes), one capture per cell.
#   FLEETQ_OWNER=claude-kvonce fleetq run --host spark2 --quiet gpu,cpu --mem 55 --expect N --label "kvonce nsys" -- bash scripts/kvonce/nsys.sh <wname> "<extra serve args>" <cell>...
# SERVE_ENV="A=1 ..." sets the server environment.
# cell = <temperature>:<size>   e.g. 0.7:65536 0:65536   (2 reps each, the first warms nothing special: the capture covers the whole sampled.py run of that cell)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
WN=$1; EXTRA=$2; shift 2; CELLS="$*"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-20-kvonce/raw/nsys-$WN; mkdir -p "$OUT"; S=$REPO/scripts/ab35b; SC=$REPO/scripts/kvonce
HOST=spark2; HN=spark-fa00; PORT=8192; MODEL=guidance; SESS=kvonce$WN
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN='~/intarsiaml-kvonce/bin/intarsia'; W='~/intarsiaml-kvonce/out'; N="$W/nsys-$WN"
SERVE_ARGS="--backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name $MODEL --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on $EXTRA"
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
SPID=""; WPID=""
stop_all() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  rsh "nsys shutdown --session=$SESS --kill sigterm 2>/dev/null; true"
  SPID=$(rsh "pgrep -f 'bin/intarsia serve.*--port $PORT' | head -1")
  [[ -n "$SPID" ]] && rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 25); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; true"
}
trap 'stop_all; log "nsys window end"' EXIT; trap 'exit 130' INT TERM
rsh "mkdir -p $N && rm -f $W/mem-nsys-$WN.txt"
scp -q -o BatchMode=yes "$SC/memwatch.sh" "$HOST:intarsiaml-kvonce/out/"
rsh "ss -ltn | grep -q ':$PORT '" && { log "port busy"; exit 1; }
log "serve under nsys: ${SERVE_ENV:-} $BIN serve $SERVE_ARGS"
rsh "nohup env ${SERVE_ENV:-} nsys launch --session-new=$SESS --trace=cuda --cuda-graph-trace=node $BIN serve $SERVE_ARGS > $W/serve-nsys-$WN.log 2>&1 < /dev/null &"
sleep 5
PID=""; for i in $(seq 1 20); do PID=$(rsh "pgrep -f 'bin/intarsia serve.*--port $PORT' | head -1"); [[ -n $PID ]] && break; sleep 1; done
[[ -n $PID ]] || { log "no server pid"; exit 1; }
WPID=$(rsh "nohup bash $W/memwatch.sh $PID $W/mem-nsys-$WN.txt > /dev/null 2>&1 < /dev/null & echo \$!")
up=0; for i in $(seq 1 240); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $PID" 2>/dev/null || break; sleep 2; done
[[ $up == 1 ]] || { log "serve did not come up"; rsh "tail -20 $W/serve-nsys-$WN.log" | tee -a "$OUT/window.log"; exit 1; }
log "serve up (pid $PID); MemAvailable MB $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo")"
for c in $CELLS; do
  IFS=: read -r T SZ <<< "$c"; name="nsys-$WN-T$T-$SZ"
  rsh "nsys start --session=$SESS --sample=none --cpuctxsw=none --force-overwrite=true -o $N/$name" || { log "nsys start failed"; break; }
  ( cd /opt/code/deepeval && uv run python "$S/sampled.py" --url "http://$HN:$PORT/v1" --model $MODEL --arm "$name" --out "$OUT" --metrics "http://$HN:$PORT/metrics" --sizes "$SZ" --reps 2 --temperature "$T" ) > "$OUT/$name-sampled.out" 2>&1
  rsh "nsys stop --session=$SESS" > "$OUT/$name-stop.out" 2>&1; tail -3 "$OUT/$name-sampled.out" | tee -a "$OUT/window.log"
  rsh "cd $N && nsys stats --force-export=true -r cuda_gpu_kern_sum,cuda_api_sum -f csv -o $name $name.nsys-rep > /dev/null 2>&1; ls -la $N | tail -5"
done
rsh "cat $W/serve-nsys-$WN.log" > "$OUT/serve.log"; log "min MemAvailable MB: $(rsh "sort -k2 -n $W/mem-nsys-$WN.txt | head -1")"
