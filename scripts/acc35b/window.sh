#!/usr/bin/env bash
# One vLLM-down window on spark1 for the 35B accuracy work (docs/results/2026-10-14-35b-accuracy.md); a copy of scripts/prefill-cuda-window.sh.
#
#   scripts/acc35b/window.sh <max-minutes> "<label>" <script on spark1>
#
# Takes ONE exclusive fleetq lease that takes `guidance` down, stops the vLLM container, waits for its memory, runs the
# script on spark1 (with FLEETQ_JOB set, under `timeout <max-minutes>m`), and then ALWAYS (trap, same order as
# scripts/phase1b-acceptance.sh): kills our own leftover processes, starts the container, waits until
# /v1/models answers, releases the lease. Appends one line per window to $DOWNTIME_LOG (stop -> serving again).
# Never --force, never --no-queue. Hard cap: 17 minutes of script, 20 minutes of vLLM downtime per window. FLEETQ_JOB prefixes every ssh, cleanup included.
#
# Env: HOST (spark1), VLLM_CONTAINER (missioncntrl-guidance), VLLM_HOST (100.120.88.104), MEM_LEASE_GB (100),
#      DOWNTIME_LOG (docs/results/2026-10-14-35b-accuracy/downtime.log), DRY_RUN=1 (print the ssh/docker/fleetq commands only).
set -euo pipefail
(( $# == 3 )) || { sed -n 2,14p "$0"; exit 2; }
MAXMIN=$1; LABEL=$2; SCRIPT=$3
(( MAXMIN >= 1 && MAXMIN <= 17 )) || { echo "max-minutes must be 1..17 (the 20-minute window includes the vLLM restart)" >&2; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST=${HOST:-spark1}
VLLM_CONTAINER=${VLLM_CONTAINER:-missioncntrl-guidance}
VLLM_HOST=${VLLM_HOST:-100.120.88.104}
MEM_LEASE_GB=${MEM_LEASE_GB:-100}
DRY_RUN=${DRY_RUN:-0}
DOWNTIME_LOG=${DOWNTIME_LOG:-$REPO/docs/results/2026-10-14-35b-accuracy/downtime.log}
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'   # non-interactive ssh has neither cargo nor nvcc

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
run() { if [[ $DRY_RUN == 1 ]]; then log "DRY: $*"; else "$@"; fi; }
rsh() { if [[ $DRY_RUN == 1 ]]; then log "DRY ssh $HOST: $*"; else FLEETQ_JOB=$JOB ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$JOB; $*"; fi; }
rsh_out() { if [[ $DRY_RUN == 1 ]]; then echo "${FAKE_OUT:-}"; else FLEETQ_JOB=$JOB ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$JOB; $*"; fi; }

JOB=""; STOPPED=0; T_STOP=0
unset FLEETQ_JOB

mem_avail_gb() { echo $(( $(rsh_out "awk '/MemAvailable/ {print \$2}' /proc/meminfo" || echo 0) / 1024 / 1024 )); }

vllm_up() {
  [[ $DRY_RUN == 1 ]] && return 0
  curl -sf -m 5 "http://$VLLM_HOST:8000/v1/models" 2>/dev/null | grep -q '"id"'
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  [[ $rc -ne 0 ]] && log "exiting with status $rc, restoring the box"
  if [[ $STOPPED == 1 ]]; then
    log "killing leftover processes of ours under ~/intarsia-acc35b"
    # driver scripts first (else the next step of a still-running script starts while vLLM loads), then binaries, nsys
    rsh "pkill -TERM -f '[i]ntarsia-acc35b/scripts/'; pkill -TERM -f '[n]sys profile'; sleep 1; pkill -KILL -f '[i]ntarsia-acc35b/scripts/'; pkill -TERM -f '[i]ntarsia-acc35b/.*/release/'; for i in \$(seq 1 20); do pgrep -f '[i]ntarsia-acc35b/.*/release/' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsia-acc35b/.*/release/'; pkill -KILL -f '[n]sys profile'; sleep 2; true"
    log "starting $VLLM_CONTAINER and waiting for /v1/models"
    rsh "FLEETQ_JOB=$JOB docker start $VLLM_CONTAINER >/dev/null"
    local ok=0 i
    for i in $(seq 1 240); do                      # up to 20 minutes
      if vllm_up; then ok=1; break; fi
      sleep 5
    done
    local t_up; t_up=$(date +%s)
    if [[ $ok == 1 ]]; then log "vLLM is serving again"; else log "WARNING: vLLM did not answer /v1/models within 20 minutes; tell the owner of guidance"; fi
    local dur=$(( t_up - T_STOP ))
    printf '%s job=%s stop=%s up=%s down=%d.%d min label="%s" serving=%s\n' "$(date +%F)" "$JOB" \
      "$(date -r "$T_STOP" +%H:%M:%S 2>/dev/null || date -d "@$T_STOP" +%H:%M:%S)" "$(date +%H:%M:%S)" $((dur / 60)) $(( (dur % 60) * 10 / 60 )) \
      "$LABEL" "$ok" >> "$DOWNTIME_LOG"
  fi
  if [[ -n "$JOB" ]]; then log "releasing fleetq job $JOB"; run fleetq release "$JOB" >/dev/null || log "WARNING: release of $JOB failed"; fi
  log "done (status $rc)"
  exit $rc
}
trap cleanup EXIT INT TERM

log "fleetq status:"; run fleetq status --host "$HOST" | head -8 || true
if [[ $DRY_RUN == 1 ]]; then JOB=9999; else
  JOB=$(fleetq acquire --host "$HOST" --exclusive --takes-down guidance --mem "$MEM_LEASE_GB" --expect "$MAXMIN" --label "$LABEL" --wait | grep -Eo '[0-9]+' | tail -1)
fi
[[ -n "$JOB" ]] || { log "could not read the fleetq job id"; exit 1; }
log "lease $JOB held"

STOPPED=1; T_STOP=$(date +%s)                      # set first: if docker stop half-fails the trap still tries docker start
rsh "FLEETQ_JOB=$JOB docker stop $VLLM_CONTAINER >/dev/null"
if [[ $DRY_RUN != 1 ]]; then
  ok=0
  for i in $(seq 1 60); do
    gb=$(mem_avail_gb)
    (( gb >= 70 )) && { log "memory freed: ${gb} GB available"; ok=1; break; }
    sleep 5
  done
  [[ $ok == 1 ]] || { log "MemAvailable did not recover after stopping vLLM"; exit 1; }
fi

log "running $SCRIPT (cap ${MAXMIN} min)"
rc=0
rsh "FLEETQ_JOB=$JOB timeout ${MAXMIN}m bash $SCRIPT" || rc=$?
log "script exit $rc"
exit $rc
