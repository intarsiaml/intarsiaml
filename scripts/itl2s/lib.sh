#!/usr/bin/env bash
# itl2s shared helpers (sourced). Runs on the Mac under `fleetq run --host spark2`, drives spark2 over ssh. bash 3.2 safe.
# One `intarsia serve` of this lane at a time on :8194 (35B NVFP4, --max-batch 2), watched by memwatch.sh (kills under 10 GB available).
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=${OUT:-$REPO/docs/results/2026-11-16-itl2s/raw}; mkdir -p "$OUT"
S=$REPO/scripts/itl2s
HOST=${HOST:-spark2}; HN=${HN:-spark-fa00}; PORT=8194; VPORT=8000
L='~/intarsia-itl2s'
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
MS=32768; MB=40
SPID=""; WPID=""; T0=$(date +%s); LOG=$OUT/${LOGNAME_:-run}.log
rsh() { ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=${FLEETQ_JOB:-none}; $*"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$LOG"; }
stop_server() {
  [ -n "$WPID" ] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [ -n "$SPID" ] && { rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 25); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID; sleep 3"; SPID=""; }
}
cleanup() { trap - EXIT INT TERM; stop_server; log "end"; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
# arm_setup <visit>: sets BIN (remote path), ENVS, ARGS from the arm name before the first dot
#   F    recipe defaults (batched MTP N=2, batched verify, verify top-k, draft vocab 131072)
#   FB0  F with INTARSIA_BATCHED_VERIFY=0 (per-sequence verify)
#   OLD  main's defaults before #386 (INTARSIA_BATCHED_MTP=1, BATCHED_VERIFY=0, VERIFY_TOPK=0, --mtp-draft-vocab 0)
#   N1   F with INTARSIA_BATCHED_MTP=1 only
#   OFF  --spec off (floor)
#   X*   experiment arms: ARM_ENV / ARM_ARGS passed in the environment (X1.a ...), binary ARM_BIN (default the lane binary)
arm_setup() {
  BIN=$L/bin/intarsia-itl2s; ENVS="X=1"; ARGS=""
  case ${1%%.*} in
    F) ;;
    FB0) ENVS="INTARSIA_BATCHED_VERIFY=0" ;;
    OLD) ENVS="INTARSIA_BATCHED_MTP=1 INTARSIA_BATCHED_VERIFY=0 INTARSIA_VERIFY_TOPK=0"; ARGS="--mtp-draft-vocab 0" ;;
    N1) ENVS="INTARSIA_BATCHED_MTP=1" ;;
    OFF) ARGS="--spec off" ;;
    X*) ENVS="X=1 ${ARM_ENV:-}"; ARGS="${ARM_ARGS:-}"; BIN=${ARM_BIN:-$BIN} ;;
    *) return 1 ;;
  esac
}
# start_serve <visit> "<ENV=..>" <extra args...>
start_serve() {
  local name=$1 envs=$2; shift 2
  log "serve $name: env [$envs INTARSIA_CB_TRACE=1] $BIN serve --backend cuda --max-seq $MS --mem-budget $MB --max-batch 2 $*"
  rsh "mkdir -p $L/out; rm -f $L/out/mem-$name.txt"
  SPID=$(rsh "nohup env $envs INTARSIA_CB_TRACE=1 $BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq $MS --mem-budget $MB --profile-every 0 --max-batch 2 $* > $L/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash $L/memwatch.sh $SPID $name > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 240); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [ $up = 1 ] || { log "serve $name did not come up"; rsh "tail -20 $L/out/serve-$name.log" | tee -a "$LOG"; stop_server; return 1; }
  log "serve $name up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded |batched MTP|draft|batched verify' $L/out/serve-$name.log | cut -c1-200" | tr '\n' ';')"
}
finish_serve() { # finish_serve <visit>
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$1.metrics" 2>/dev/null
  rsh "gzip -c $L/out/serve-$1.log" > "$OUT/serve-$1.log.gz"
  log "min MemAvailable MB (time MB; KILLED count) ($1): $(rsh "sort -k2 -n $L/out/mem-$1.txt | head -1; grep -c KILLED $L/out/mem-$1.txt" | tr '\n' ' ')"
  stop_server
}
vllm_busy() { curl -s --max-time 4 "http://$HN:$VPORT/metrics" | awk '/^vllm:num_requests_running/ {s+=$NF} END {print s+0}'; }
# run_cells <visit>: the client on the Spark for each cell of $CELLS ("set:temp:thinking ..."), results copied to $OUT/cells/
run_cells() {
  local v=$1 c set temp th
  # 1 Hz sample of the neighbour's load: vLLM requests running / waiting on spark2, with the epoch second
  rsh "rm -f $L/out/vllm-$v.txt; nohup bash $L/scripts/itl2s/vsample.sh $L/out/vllm-$v.txt > /dev/null 2>&1 < /dev/null & echo \$! > $L/out/vllm-$v.pid"
  for c in ${CELLS:?CELLS}; do
    set=${c%%:*}; c=${c#*:}; temp=${c%%:*}; th=${c#*:}
    rsh "python3 $L/scripts/itl2s/itl.py --url http://localhost:$PORT/v1 --model guidance --visit $v --set $set --temperature $temp --thinking $th --streams ${STREAMS:-2} --rounds ${ROUNDS:-2} --gen ${GEN:-384} --out $L/out/cells" 2>&1 | tee -a "$LOG"
  done
  rsh "kill \$(cat $L/out/vllm-$v.pid) 2>/dev/null; gzip -c $L/out/vllm-$v.txt" > "$OUT/vllm-$v.txt.gz"
  mkdir -p "$OUT/cells"; ssh -o BatchMode=yes "$HOST" "cd $L/out/cells && tar cf - itl-$v-*.json" | tar xf - -C "$OUT/cells"
}
