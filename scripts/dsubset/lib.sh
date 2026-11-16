#!/usr/bin/env bash
# dsubset shared helpers (sourced by the run scripts; adapted from scripts/dvocab/lib.sh). Everything runs inside a fleetq slot
# on spark1 (FLEETQ_JOB set). spark1 vLLM (flight, guidance, capcom) stays up; one `intarsia serve` of this lane at a time on :8197, watched by
# memwatch.sh (kill under 10 GB available).
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=${OUT:-$REPO/docs/results/2026-11-15-dsubset/raw/runs}; mkdir -p "$OUT"
S=$REPO/scripts/dsubset
HOST=${HOST:-spark1}; case $HOST in spark1) HN=spark-4e77 ;; spark2) HN=spark-fa00 ;; esac; PORT=8197; VPORT=8000
M35=${M35:-/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5}
BIN=${BIN:-'~/intarsia-dsubset/bin/intarsia'}; MS=32768; MB=40
SPID=""; WPID=""; T0=$(date +%s); LOG=$OUT/${LOGNAME_:-run}.log
rsh() { ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$LOG"; }
stop_server() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 25); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID; sleep 3"; SPID=""; }
}
cleanup() { trap - EXIT INT TERM; stop_server; log "end"; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
# start_serve <name> "<ENV=.. ENV=..>" <extra args...>
start_serve() {
  local name=$1 envs=$2; shift 2
  log "serve arm $name: env [$envs] serve --backend cuda --model <HF snapshot 1355db6a> --max-seq $MS --mem-budget $MB --profile-every 0 $*"
  rsh "rm -f ~/intarsia-dsubset/out/mem-$name.txt"
  SPID=$(rsh "nohup env $envs $BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq $MS --mem-budget $MB --profile-every 0 $* > ~/intarsia-dsubset/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash ~/intarsia-dsubset/memwatch.sh $SPID $name > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 180); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 ~/intarsia-dsubset/out/serve-$name.log" | tee -a "$LOG"; stop_server; return 1; }
  log "serve $name up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded |batched MTP|draft' ~/intarsia-dsubset/out/serve-$name.log | cut -c1-200" | tr '\n' ';')"
}
finish_serve() { # finish_serve <name>
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$1.metrics" 2>/dev/null
  rsh "cat ~/intarsia-dsubset/out/serve-$1.log" > "$OUT/serve-$1.log"
  log "min MemAvailable MB (time MB; KILLED count) ($1): $(rsh "sort -k2 -n ~/intarsia-dsubset/out/mem-$1.txt | head -1; grep -c KILLED ~/intarsia-dsubset/out/mem-$1.txt" | tr '\n' ' ')"
  stop_server
}
# requests another client has in flight on spark1 vLLM (noise for a timing run)
vllm_busy() { curl -s --max-time 4 "http://$HN:$VPORT/metrics" | awk '/^vllm:num_requests_running/ {s+=$NF} END {print s+0}'; }
URL=http://$HN:$PORT/v1
