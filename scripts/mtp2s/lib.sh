#!/usr/bin/env bash
# mtp2s shared helpers (sourced by ab.sh / gates.sh; adapted from scripts/mtp-sampled/lib.sh). Everything runs inside a fleetq slot
# on spark2 (FLEETQ_JOB set). spark2's vLLM (retro) stays up; one `intarsia serve` of this lane at a time on :8194, watched by
# memwatch.sh (kill under 10 GB available).
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=${OUT:-$REPO/docs/results/2026-11-14-mtp2s/raw/ab}; mkdir -p "$OUT"
S=$REPO/scripts/mtp2s
HOST=spark2; HN=spark-fa00; PORT=8194; VPORT=8000
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN=${BIN:-'~/intarsia-mtp2s/bin/intarsia'}; MS=32768; MB=40
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
  rsh "rm -f ~/intarsia-mtp2s/out/mem-$name.txt"
  SPID=$(rsh "nohup env $envs $BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq $MS --mem-budget $MB --profile-every 0 $* > ~/intarsia-mtp2s/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash ~/intarsia-mtp2s/memwatch.sh $SPID $name > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 180); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 ~/intarsia-mtp2s/out/serve-$name.log" | tee -a "$LOG"; stop_server; return 1; }
  log "serve $name up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded |batched MTP|draft' ~/intarsia-mtp2s/out/serve-$name.log | cut -c1-200" | tr '\n' ';')"
}
finish_serve() { # finish_serve <name>
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$1.metrics" 2>/dev/null
  rsh "cat ~/intarsia-mtp2s/out/serve-$1.log" > "$OUT/serve-$1.log"
  log "min MemAvailable MB (time MB; KILLED count) ($1): $(rsh "sort -k2 -n ~/intarsia-mtp2s/out/mem-$1.txt | head -1; grep -c KILLED ~/intarsia-mtp2s/out/mem-$1.txt" | tr '\n' ' ')"
  stop_server
}
# requests another client has in flight on spark2's vLLM (noise for a timing run)
vllm_busy() { curl -s --max-time 4 "http://$HN:$VPORT/metrics" | awk '/^vllm:num_requests_running/ {s+=$NF} END {print s+0}'; }
URL=http://$HN:$PORT/v1
