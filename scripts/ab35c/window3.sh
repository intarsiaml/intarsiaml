#!/usr/bin/env bash
# ab35c window 3 (2-stream store): spark2 vLLM-down window for cutover blockers 1, 3, 5 (docs/results/2026-11-12-ab35b-cutover-followup.md).
# usage: window.sh <cap-minutes>
# Runs INSIDE: FLEETQ_OWNER=claude-ab35c fleetq run --host spark2 --takes-down retro --quiet gpu,cpu --mem 58 --expect N ...
# Stops spark2's vLLM, runs the arms one `intarsia serve` at a time, restores vLLM on ANY exit (trap) and waits for
# /v1/models 200. Hard cap on vLLM downtime = <cap-minutes>; no new arm starts after 80 % of it.
#   C-default  (PR #351 build, --max-batch 2): tool smoke thinking off (blocker 1 receipt turns) + thinking default (stock client)
#   C-fp8      (PR #351 build, --max-batch 2 --kv-dtype fp8): sampled 32k/64k (blocker 5 first step)
#   C-store    (PR #351 build, --prefix-store on, INTARSIA_PREFIX_STORE_DEBUG=1): replay interleaved seed 0 (blocker 3, before)
#   C-storefix (store-settle-off-decode build, same flags): replay interleaved seed 0 (blocker 3, after)
#   C-single-off (PR #351 build, single engine, INTARSIA_SPEC_SAMPLED=0): sampled 4k/32k/64k (the missing --spec off control)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
CAPMIN=${1:-55}
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
HOST=spark2; PORT=8191; VPORT=8000; CT=missioncntrl-guidance
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN351='~/intarsia-ab35b/bin/intarsia'; BINFIX='~/intarsia-ab35c-fix/bin/intarsia'; MS=131072; MB=40
CAPSEC=$((CAPMIN*60)); SOFT=$((CAPSEC*8/10))
SPID=""; WPID=""; DOWN=0; DOWN_T=0; T0=$(date +%s); CAPPID=""
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$OUT/window3.log"; }
stop_server() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 25); do pgrep -f 'bin/intarsia serve' >/dev/null || exit 0; sleep 1; done; pkill -9 -f 'bin/intarsia serve'; sleep 3"; SPID=""; }
}
restore() {
  trap - EXIT INT TERM; [[ -n "$CAPPID" ]] && kill $CAPPID 2>/dev/null
  stop_server
  if [[ $DOWN == 1 ]]; then
    log "restoring vLLM: docker start $CT"
    rsh "docker start $CT" 2>&1 | tee -a "$OUT/window3.log"
    local i; for i in $(seq 1 150); do
      [[ $(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://$HN:$VPORT/v1/models") == 200 ]] && { log "vLLM /v1/models 200 at $(date +%T) (vLLM down for $(( $(date +%s)-DOWN_T ))s total)"; DOWN=0; break; }
      sleep 4
    done
    [[ $DOWN == 1 ]] && log "WARNING: vLLM did not answer /v1/models within 10 min after docker start"
  fi
  log "window end"
}
trap restore EXIT; trap 'restore; exit 130' INT TERM
start_serve() { # start_serve <name> <bin> <env-assignments or -> <serve args...>
  local name=$1 bin=$2 envs=$3; shift 3; [[ $envs == - ]] && envs=""
  log "serve arm $name: ${envs:+$envs }$bin serve --backend cuda --model <HF snapshot 1355db6a> --port $PORT --max-seq $MS --mem-budget $MB --profile-every 0 $*"
  rsh "rm -f ~/intarsia-ab35b/out/mem-$name.txt; mkdir -p ~/intarsia-ab35b/out"
  SPID=$(rsh "nohup env $envs $bin serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq $MS --mem-budget $MB --profile-every 0 $* > ~/intarsia-ab35b/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash ~/intarsia-ab35b/memwatch.sh $SPID $name > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 180); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 ~/intarsia-ab35b/out/serve-$name.log" | tee -a "$OUT/window3.log"; stop_server; return 1; }
  log "serve $name up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded |sampled requests' ~/intarsia-ab35b/out/serve-$name.log | tr '\n' ' '")"
}
finish_serve() { # finish_serve <name>
  sleep 3  # store settle now runs after Done: let the last one land before the scrape
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$1.metrics" 2>/dev/null
  rsh "cat ~/intarsia-ab35b/out/serve-$1.log" > "$OUT/serve-$1.log"
  rsh "sort -k2 -n ~/intarsia-ab35b/out/mem-$1.txt | head -1; grep -c KILLED ~/intarsia-ab35b/out/mem-$1.txt" | tr '\n' ' ' | sed "s/^/min MemAvailable MB (time MB; KILLED count) ($1): /" | tee -a "$OUT/window3.log"; echo | tee -a "$OUT/window3.log" >/dev/null
  stop_server
}
late() { (( $(date +%s)-T0 > SOFT )); }
URL=http://$HN:$PORT/v1
log "window start (cap ${CAPMIN} min); vLLM up: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom')"

scp -q -o BatchMode=yes "$S/memwatch.sh" "$HOST:intarsia-ab35b/memwatch.sh" || { log "memwatch copy failed; aborting before vLLM stop"; exit 1; }
rsh "docker inspect $CT --format '{{.State.StartedAt}} {{.Config.Image}}'" | tee -a "$OUT/window3.log"
rsh "docker stop $CT" 2>&1 | tee -a "$OUT/window3.log"; DOWN=1; DOWN_T=$(date +%s)
log "vLLM DOWN (docker stop done) at $(date +%T)"
(sleep $CAPSEC; echo "[cap] ${CAPMIN} min down reached, restoring" >> "$OUT/window3.log"; kill -TERM $$) >/dev/null 2>&1 &
CAPPID=$!
rsh "free -g | sed -n 2p; nvidia-smi --query-gpu=memory.used --format=csv,noheader" | tee -a "$OUT/window3.log"

# window 3: 2-stream replay (replay2 --streams 2): batched no store vs batched + store (#353); #354 single-engine replay
BIN353='~/intarsia-w3-353/bin/intarsia'; BIN354='~/intarsia-w3-354/bin/intarsia'
log "binaries: 353 $(rsh "$BIN353 --version | head -1"); 354 $(rsh "$BIN354 --version | head -1")"
if start_serve D-nostore "$BIN353" - --max-batch 2; then
  replay2 D-nostore "$URL" interleaved 0 2
  finish_serve D-nostore
fi
if ! late && start_serve D-store "$BIN353" INTARSIA_PREFIX_STORE_DEBUG=1 --max-batch 2 --prefix-store on; then
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-D-store.metrics.before" 2>/dev/null
  replay2 D-store "$URL" interleaved 0 2
  finish_serve D-store
else log "skipping D-store"; fi
if ! late && start_serve C-storecap "$BIN354" INTARSIA_PREFIX_STORE_DEBUG=1 --prefix-store on; then
  replay C-storecap "$URL" interleaved 0
  finish_serve C-storecap
else log "skipping C-storecap"; fi
