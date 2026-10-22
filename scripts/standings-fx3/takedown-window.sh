#!/usr/bin/env bash
# fx3: ONE vLLM-down window on spark1 for the arm that cannot share the box with the production vLLM (SGLang 35B NVFP4 needs ~48 GB).
#   fleetq run --host spark1 --exclusive --takes-down flight,guidance,capcom --mem 100 --expect 32 --label "..." -- \
#       scripts/standings-fx3/takedown-window.sh "r21:sglang r22:intarsia r23:sglang r24:intarsia r25:sglang"
# Each entry is <pass id>:<arm>, run through window.sh (MODEL=35b, speed+conc). The EXIT trap restarts the production container
# (docker start, same as scripts/prefill-cuda-window.sh), waits until /v1/models answers and appends the downtime to raw/downtime.log.
# Caps (lane rules): windows <= 30 min of serving down, >= 5 min serving between windows, 00:00-07:00 EDT the daily cap is lifted.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPEC=${1:?"pass:arm ..."}
HOST=${HOST:-spark1}; VLLM_CONTAINER=${VLLM_CONTAINER:-missioncntrl-guidance}; VLLM_HOST=${VLLM_HOST:-100.120.88.104}
OUT=$REPO/docs/results/2026-10-22-fx3-spark-standings/raw/35b
LOG=$REPO/docs/results/2026-10-22-fx3-spark-standings/raw/downtime.log
rsh() { ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { echo "[$(date +%T)] $*"; }
vllm_up() { curl -sf -m 5 "http://$VLLM_HOST:8000/v1/models" >/dev/null 2>&1; }
STOPPED=0; T_STOP=0
restore() {
  trap - EXIT INT TERM
  kill "$WD" 2>/dev/null; [[ -n "${WPID:-}" ]] && kill "$WPID" 2>/dev/null && sleep 5
  if (( STOPPED )); then
    rsh "docker rm -f fx3-sglang >/dev/null 2>&1; for p in \$(pgrep -f 'intarsia-decfuse/fx3-main/.*/intarsia serve'); do kill \$p; done; sleep 3; true" >/dev/null 2>&1
    log "starting $VLLM_CONTAINER, waiting for /v1/models"
    rsh "docker start $VLLM_CONTAINER >/dev/null"
    local ok=0 i
    for i in $(seq 1 240); do if vllm_up; then ok=1; break; fi; sleep 5; done
    local t_up; t_up=$(date +%s); local dur=$(( t_up - T_STOP ))
    (( ok )) && log "vLLM serving again" || log "WARNING: vLLM did not answer within 20 minutes; tell owner guidance"
    printf '%s job=%s stop=%s up=%s down=%d.%d min label="fx3 spark1 SGLang 35B window" serving=%s\n' "$(date +%F)" "$FLEETQ_JOB" \
      "$(date -r "$T_STOP" +%H:%M:%S 2>/dev/null || date -d "@$T_STOP" +%H:%M:%S)" "$(date +%H:%M:%S)" $((dur / 60)) $(( (dur % 60) * 10 / 60 )) "$ok" >> "$LOG"
  fi
}
trap restore EXIT INT TERM
# Watchdog: the window must not outlast the 30-minute cap; TERM runs the restore trap.
( sleep "${MAXSEC:-1500}"; kill -TERM $$ ) >/dev/null 2>&1 &
WD=$!
vllm_up || { log "ABORT: production vLLM not answering before the window"; STOPPED=0; exit 3; }
STOPPED=1; T_STOP=$(date +%s)
rsh "docker stop $VLLM_CONTAINER >/dev/null"
for i in $(seq 1 30); do
  avail=$(rsh "awk '/MemAvailable/ {print int(\$2/1024/1024)}' /proc/meminfo"); (( avail >= 90 )) && break; sleep 4
done
log "vLLM stopped; MemAvailable ${avail:-?} GB"
for e in $SPEC; do
  id=${e%%:*}; arm=${e#*:}
  log "== $id $arm"
  HOST=$HOST MACHINE=spark-4e77 MODEL=35b PASS=$id ARMS=$arm PHASES=${WPHASES:-speed,conc} MINGB=80 SGL_MEMFRAC=${SGL_MEMFRAC:-0.50} OUT=$OUT \
    "$REPO/scripts/standings-fx3/window.sh" &
  WPID=$!; wait $WPID
done
log "window done"
