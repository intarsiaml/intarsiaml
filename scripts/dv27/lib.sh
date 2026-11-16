#!/usr/bin/env bash
# dv27 shared helpers (adapted from scripts/dvocab/lib.sh). Everything runs inside a fleetq slot on spark1 (FLEETQ_JOB set). spark1's vLLM
# (flight / guidance / capcom) stays up; one `intarsia serve` of this lane at a time on :8195, watched by memwatch.sh (kill under 10 GB available).
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=${OUT:-$REPO/docs/results/2026-11-15-dv27/raw/sweep}; mkdir -p "$OUT"
S=$REPO/scripts/dv27
HOST=spark1; HN=spark-4e77; PORT=8195; VPORT=8000
M27=/home/nvidia/models/Qwen3.8-27B-Quark-AWQ-MXFP4
BIN=${BIN:-'~/intarsia-dv27/bin/intarsia'}; MS=${MS:-8192}; MB=${MB:-30}
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
  log "serve arm $name: env [$envs] serve --backend cuda --model $M27 --max-seq $MS --mem-budget $MB --profile-every 0 $*"
  rsh "rm -f ~/intarsia-dv27/out/mem-$name.txt"
  SPID=$(rsh "nohup env $envs $BIN serve --backend cuda --model $M27 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq $MS --mem-budget $MB --profile-every 0 $* > ~/intarsia-dv27/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash ~/intarsia-dv27/memwatch.sh $SPID $name > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 240); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 ~/intarsia-dv27/out/serve-$name.log" | tee -a "$LOG"; stop_server; return 1; }
  log "serve $name up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded |batched MTP|draft|mtp' ~/intarsia-dv27/out/serve-$name.log | cut -c1-200" | tr '\n' ';')"
}
finish_serve() { # finish_serve <name>
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$1.metrics" 2>/dev/null
  rsh "cat ~/intarsia-dv27/out/serve-$1.log" > "$OUT/serve-$1.log"
  log "min MemAvailable MB (time MB; KILLED count) ($1): $(rsh "sort -k2 -n ~/intarsia-dv27/out/mem-$1.txt | head -1; grep -c KILLED ~/intarsia-dv27/out/mem-$1.txt" | tr '\n' ' ')"
  stop_server
}
# requests another client has in flight on spark1's vLLM (noise for a timing run)
vllm_busy() { curl -s --max-time 4 "http://$HN:$VPORT/metrics" | awk '/^vllm:num_requests_running/ {s+=$NF} END {print s+0}'; }
URL=http://$HN:$PORT/v1
