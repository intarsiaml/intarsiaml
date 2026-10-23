#!/usr/bin/env bash
# The measurement window of this lane, run on endor INSIDE a fleetq slot that takes the box's vLLM role down:
#   fleetq acquire --host spark2 --takes-down retro --quiet gpu,cpu --expect 90 --owner claude-f31ccf91-mtp --wait
#   FLEETQ_JOB=<id> HOST=spark2 MACHINE=spark-fa00 REF=<sha> VLLM_CONTAINER=<name> bash .../excl-window.sh
# Stops the vLLM container, runs the per-pass phase timings and three interleaved localeval passes (off, k=1..3), and
# restarts the container in an EXIT trap, waiting until /v1/models answers. Downtime goes to raw/downtime.log.
set -uo pipefail
: "${FLEETQ_JOB:?}"; REF=${REF:?}; HOST=${HOST:?}; MACHINE=${MACHINE:?}; VLLM_CONTAINER=${VLLM_CONTAINER:?}
D=docs/results/2026-10-22-mtp-spec-decode
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
mkdir -p $D/raw
rsh() { ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { echo "[$(date +%T)] $*"; }
vllm_up() { curl -sf -m 5 "http://$MACHINE:8000/v1/models" >/dev/null 2>&1; }
STOPPED=0; T_STOP=0
restore() {
  trap - EXIT INT TERM
  [[ -n "${WPID:-}" ]] && kill "$WPID" 2>/dev/null
  rsh 'p=$(cat $HOME/mtp-run/server.pid 2>/dev/null); [ -n "$p" ] && kill $p 2>/dev/null; sleep 3; true'
  if (( STOPPED )); then
    log "starting $VLLM_CONTAINER, waiting for /v1/models"
    rsh "docker start $VLLM_CONTAINER >/dev/null"
    local ok=0 i
    for i in $(seq 1 240); do if vllm_up; then ok=1; break; fi; sleep 5; done
    local dur=$(( $(date +%s) - T_STOP ))
    (( ok )) && log "vLLM serving again" || log "WARNING: vLLM did not answer within 20 minutes"
    printf '%s job=%s host=%s role=retro stop=%s up=%s down=%d min serving=%s\n' "$(date +%F)" "$FLEETQ_JOB" "$HOST" \
      "$(date -r "$T_STOP" +%H:%M:%S)" "$(date +%H:%M:%S)" $((dur / 60)) "$ok" >> $D/raw/downtime.log
  fi
}
trap restore EXIT INT TERM
vllm_up || { log "ABORT: $HOST vLLM not answering before the window"; exit 3; }
STOPPED=1; T_STOP=$(date +%s)
rsh "docker stop $VLLM_CONTAINER >/dev/null"
for i in $(seq 1 30); do
  avail=$(rsh "awk '/MemAvailable/ {print int(\$2/1024/1024)}' /proc/meminfo"); (( avail >= 90 )) && break; sleep 4
done
log "vLLM stopped; MemAvailable ${avail:-?} GB"
scripts/deploy.sh "$HOST" --backend cuda --ref "$REF" --tree mtp --smoke quick --no-build -- bash $D/scripts/phases.sh target/release/intarsia replay-out/mtp-phases
rsync -a "$HOST:intarsia-deploy/mtp/replay-out/mtp-phases/" $D/raw/phases/
for spec in "p1:off mtp1 mtp2 mtp3" "p2:mtp3 mtp2 mtp1 off" "p3:mtp2 off mtp3 mtp1"; do
  HOST=$HOST MACHINE=$MACHINE REF=$REF PASS=${spec%%:*} ARMS="${spec#*:}" bash $D/scripts/window.sh &
  WPID=$!; wait $WPID || { log "pass ${spec%%:*} failed"; exit 4; }
done
log "window done"
