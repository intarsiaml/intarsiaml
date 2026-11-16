#!/usr/bin/env bash
# mx6 shared helpers (sourced by gates.sh; adapted from scripts/flipprep/lib.sh and scripts/dv27/lib.sh). Everything runs inside a fleetq slot
# (FLEETQ_JOB set) on HOST (spark1: the 27B and the 35B; spark2: the 35B). The host's vLLM stays up; one `intarsia serve` of this lane at a
# time on :8196, watched by memwatch.sh (kill under 10 GB available).
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
HOST=${HOST:-spark1}; PORT=8196; VPORT=8000
case $HOST in spark1) HN=spark-4e77 ;; spark2) HN=spark-fa00 ;; *) echo "HOST?"; exit 2 ;; esac
MODEL=${MODEL:-27}
case $MODEL in
  27) MP=/home/nvidia/models/Qwen3.8-27B-Quark-AWQ-MXFP4; MB=30; MS=8192; TAG=27b ;;
  35) MP=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5; MB=40; MS=8192; TAG=35b ;;
esac
OUT=${OUT:-$REPO/docs/results/2026-11-15-mx6/raw/$TAG-$HOST}; mkdir -p "$OUT"
S=$REPO/scripts/mx6; L='~/intarsia-mx6'; B="$L/bin"
SPID=""; WPID=""; T0=$(date +%s); LOG=$OUT/${LOGNAME_:-run}.log
rsh() { ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$LOG"; }
stop_server() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 25); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID; sleep 3"; SPID=""; }
}
cleanup() { trap - EXIT INT TERM; stop_server; log "end"; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
# arm_setup <arm>: BIN, ENVS, ARGS. Arms (`--max-batch 2` is added by the caller):
#   R   = mx6 binary, --spec off                       (the ids reference)
#   M   = mx6 binary, the failing opt-in set           (INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 INTARSIA_VERIFY_TOPK=1, --spec mtp --spec-k 2 --mtp-draft-vocab DV)
#   MM  = main binary, the same opt-in set             (the 27B crashes at the first request; the 35B serves)
#   N   = mx6 binary, batched MTP without batched verify (INTARSIA_BATCHED_MTP=2 only)
arm_setup() {
  local dv=65536; [[ $MODEL == 35 ]] && dv=131072
  ENVS="X=1"; ARGS=""
  case ${1%%.*} in
    R)  BIN=$B/intarsia-mx6; ARGS="--spec off" ;;
    M)  BIN=$B/intarsia-mx6; ENVS="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 INTARSIA_VERIFY_TOPK=1"; ARGS="--spec mtp --spec-k 2 --mtp-draft-vocab $dv" ;;
    MM) BIN=$B/intarsia-main; ENVS="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 INTARSIA_VERIFY_TOPK=1"; ARGS="--spec mtp --spec-k 2 --mtp-draft-vocab $dv" ;;
    N)  BIN=$B/intarsia-mx6; ENVS="INTARSIA_BATCHED_MTP=2"; ARGS="--spec mtp --spec-k 2 --mtp-draft-vocab $dv" ;;
    *)  return 1 ;;
  esac
}
# start_serve <name> "<ENV=.. ENV=..>" <extra args...>
start_serve() {
  local name=$1 envs=$2; shift 2
  log "serve arm $name: env [$envs] $BIN serve --backend cuda --model <$TAG> --max-seq $MS --mem-budget $MB --profile-every 0 $*"
  rsh "rm -f $L/out/mem-$name.txt"
  SPID=$(rsh "nohup env $envs $BIN serve --backend cuda --model $MP --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq $MS --mem-budget $MB --profile-every 0 $* > $L/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash $L/memwatch.sh $SPID $name > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 240); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 $L/out/serve-$name.log" | tee -a "$LOG"; stop_server; return 1; }
  log "serve $name up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo")"
  log "load lines ($name): $(rsh "grep -E 'batched verify|batched MTP|admission on plan|speculative decoding' $L/out/serve-$name.log | cut -c1-260" | tr '\n' '|')"
}
finish_serve() { # finish_serve <name>
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$1.metrics" 2>/dev/null
  rsh "cat $L/out/serve-$1.log" > "$OUT/serve-$1.log"
  log "min MemAvailable MB (time MB; KILLED count) ($1): $(rsh "sort -k2 -n $L/out/mem-$1.txt | head -1; grep -c KILLED $L/out/mem-$1.txt" | tr '\n' ' ')"
  stop_server
}
URL=http://$HN:$PORT/v1
