#!/usr/bin/env bash
# ab35 window (runs INSIDE: fleetq run --host spark2 --takes-down retro --quiet gpu,cpu --expect 60 --mem <peak> ...):
# stop spark2 vLLM -> intarsia serve (shipped defaults, --max-batch 2) -> identical localeval suite -> `--spec off` arm (speed only)
# -> restore vLLM and wait for /v1/models 200. Restore runs on ANY exit (trap). Hard cap: vLLM down <= 60 min.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=spark2; HN=spark-fa00; PORT=8191; VPORT=8000; CT=missioncntrl-guidance
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-12-ab35-intarsia-vs-vllm/raw; mkdir -p "$OUT"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN='~/intarsia-ab35/bin/intarsia'; SN=guidance; MS=20480; MB=40
CAPSEC=$((45*60)); SOFT=$((36*60))     # hard cap on vLLM downtime; no new arm after SOFT seconds
SPID=""; WPID=""; DOWN=0; T0=$(date +%s)
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$OUT/window.log"; }
LEPID=""
run_le() { "$REPO/scripts/ab35/le.sh" "$@" & LEPID=$!; wait $LEPID; LEPID=""; }
stop_server() {
  [[ -n "$LEPID" ]] && { pkill -TERM -P $LEPID 2>/dev/null; kill $LEPID 2>/dev/null; LEPID=""; }
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 20); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 3"; SPID=""; }
}
restore() {
  trap - EXIT INT TERM; kill $CAPPID 2>/dev/null
  stop_server
  if [[ $DOWN == 1 ]]; then
    log "restoring vLLM: docker start $CT"
    rsh "docker start $CT" 2>&1 | tee -a "$OUT/window.log"
    local i; for i in $(seq 1 150); do
      [[ $(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://$HN:$VPORT/v1/models") == 200 ]] && { log "vLLM /v1/models 200 (vLLM down for $(( $(date +%s)-DOWN_T ))s total)"; DOWN=0; break; }
      sleep 4
    done
    [[ $DOWN == 1 ]] && log "WARNING: vLLM did not answer /v1/models within 10 min after docker start"
  fi
  log "window end"
}
trap restore EXIT; trap 'restore; exit 130' INT TERM
CAPPID=""
start_serve() { # start_serve <name> <extra args...>
  local name=$1; shift
  log "serve arm $name: $BIN serve $*"
  SPID=$(rsh "nohup $BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name $SN --max-seq $MS --mem-budget $MB --max-batch 2 --profile-every 0 $* > ~/intarsia-ab35/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  # memory watchdog on the box: kill the server if MemAvailable < 10 GB
  WPID=$(rsh "nohup bash ~/intarsia-ab35/memwatch.sh $SPID $name > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 180); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 ~/intarsia-ab35/out/serve-$name.log" | tee -a "$OUT/window.log"; return 1; }
  log "serve $name up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo")"
}
finish_serve() { # finish_serve <name>
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$1.metrics" 2>/dev/null
  rsh "cat ~/intarsia-ab35/out/serve-$1.log" > "$OUT/serve-$1.log"
  rsh "sort -k2 -n ~/intarsia-ab35/out/mem-$1.txt | head -1" | sed "s/^/min MemAvailable MB ($1): /" | tee -a "$OUT/window.log"
  stop_server
}
log "window start; vLLM up: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models)"
rsh "docker inspect $CT --format '{{.State.StartedAt}} {{.Config.Image}}'" | tee -a "$OUT/window.log"
rsh "docker stop $CT" 2>&1 | tee -a "$OUT/window.log"; DOWN=1; DOWN_T=$(date +%s)
log "vLLM DOWN (docker stop done)"
(sleep $CAPSEC; echo "[cap] 45 min down reached, restoring" >> "$OUT/window.log"; kill -TERM $$) >/dev/null 2>&1 &
CAPPID=$!
rsh "free -g | sed -n 2p; nvidia-smi --query-gpu=memory.used --format=csv,noheader" | tee -a "$OUT/window.log"
BASE="$BIN serve --backend cuda --model <HF snapshot 1355db6a> --port $PORT --host 0.0.0.0 --served-model-name $SN --max-seq $MS --mem-budget $MB --max-batch 2 --profile-every 0"
if start_serve B-default; then
  run_le B-default "http://$HN:$PORT/v1" $SN "~/intarsia-ab35/bin/intarsia serve ${BASE#*serve } (shipped CUDA defaults: MTP k2 via batched_mtp, bf16 KV, prefill chunk 2048, macro 8192)" all
  finish_serve B-default
  log "B-default spec counters: $(grep -E 'spec_(proposed|accepted)_tokens_total|spec_steps' "$OUT/serve-B-default.metrics" | tr '\n' ' ')"
fi
if (( $(date +%s)-T0 < SOFT )) && start_serve B-specoff --spec off; then
  run_le B-specoff "http://$HN:$PORT/v1" $SN "~/intarsia-ab35/bin/intarsia serve --spec off ${BASE#*serve }" speed
  finish_serve B-specoff
else log "skipping spec-off arm (time or start failure)"; fi
# restore runs from the trap
