#!/usr/bin/env bash
# Lane S1: one spark2 vLLM-down window (role retro), as ONE fleetq run, 00:00-07:00 EDT only (lane rules: no cap there).
# A copy of scripts/cuda-round7/window.sh with spark2's names, a 60-minute script cap and the localeval URL as LE_URL.
#   fleetq run --host spark2 --exclusive --takes-down retro --mem 100 --expect <min> --label "<label>" -- \
#     scripts/s1/window.sh <max-script-minutes> "<label>" <script on spark2>
# Stops the vLLM container, waits for the memory, runs the spark1 script (FLEETQ_JOB set, under `timeout`), and ALWAYS
# (trap on EXIT/INT/TERM): kills our leftover processes under ~/intarsia-decfuse, starts the container, waits until
# /v1/models answers, appends one line to $DOWNTIME_LOG. fleetq run releases the slot when this exits.
# LOCALEVAL_ARMS="a b": for each arm the spark1 script prints "READY <arm>"; this script then runs localeval speed against
# spark-4e77:8090 from here (FLEETQ_JOB = this window) and touches ~/intarsia-decfuse/out/$LE_OUTNAME/<arm>.done.
# Cap: script <= 22 min (window <= 30 min with the stop and the restart).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run --takes-down guidance}"
(( $# == 3 )) || { sed -n 2,11p "$0"; exit 2; }
MAXMIN=$1; LABEL=$2; SCRIPT=$3
(( MAXMIN >= 1 && MAXMIN <= 60 )) || { echo "max-minutes must be 1..60" >&2; exit 2; }
# lane rules: no cap 00:00-07:00 EDT (the cap ends the window by 07:00); outside it, windows of at most 30 min
h=$(TZ=America/New_York date +%H); mnow=$(( 10#$h * 60 + 10#$(TZ=America/New_York date +%M) ))
if (( mnow < 7 * 60 - 25 )); then
  mins_left=$(( 7 * 60 - mnow - 10 ))
  (( MAXMIN > mins_left )) && { echo "cap ${MAXMIN} min -> ${mins_left} min (07:00 EDT)"; MAXMIN=$mins_left; }
else
  (( MAXMIN > 24 )) && { echo "outside 00:00-07:00 EDT: cap ${MAXMIN} min -> 24 min (lane rules: 30 min windows incl. vLLM restart)"; MAXMIN=24; }
fi
# lane rules: agent-harness benchmarks against the Spark vLLMs; never take spark2's down under one
if fleetq status 2>&1 | grep -iq "agent-harness"; then
  echo "refusing: fleetq status shows an agent-harness job:" >&2; fleetq status 2>&1 | grep -i "agent-harness" >&2; exit 3
fi
echo "agent-harness check $(date +%T): none in fleetq status"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST=${HOST:-spark2}
VLLM_CONTAINER=${VLLM_CONTAINER:-missioncntrl-guidance}
VLLM_HOST=${VLLM_HOST:-spark-fa00}
DOWNTIME_LOG=${DOWNTIME_LOG:-$REPO/docs/results/2026-10-22-s1-spark-prefill/raw/downtime.log}
mkdir -p "$(dirname "$DOWNTIME_LOG")"
WLOG=${WLOG:-/dev/stdout}
JOB=$FLEETQ_JOB
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$JOB; $*"; }
vllm_up() { curl -sf -m 5 "http://$VLLM_HOST:8000/v1/models" 2>/dev/null | grep -q '"id"'; }
mem_avail_gb() { echo $(( $(rsh "awk '/MemAvailable/ {print \$2}' /proc/meminfo" || echo 0) / 1024 / 1024 )); }
STOPPED=0; T_STOP=0; SPID=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  [[ -n "$SPID" ]] && kill "$SPID" 2>/dev/null
  if [[ $STOPPED == 1 ]]; then
    log "killing leftover processes of ours under ~/intarsia-decfuse"
    rsh "pkill -TERM -f '[i]ntarsia-decfuse/.*/scripts/'; sleep 1; pkill -KILL -f '[i]ntarsia-decfuse/.*/scripts/'; pkill -TERM -f '[i]ntarsia-decfuse/.*/release/'; for i in \$(seq 1 20); do pgrep -f '[i]ntarsia-decfuse/.*/release/' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsia-decfuse/.*/release/'; sleep 2; true"
    log "starting $VLLM_CONTAINER and waiting for /v1/models"
    rsh "docker start $VLLM_CONTAINER >/dev/null"
    local ok=0 i
    for i in $(seq 1 240); do if vllm_up; then ok=1; break; fi; sleep 5; done
    local t_up; t_up=$(date +%s)
    if [[ $ok == 1 ]]; then log "vLLM is serving again"; else log "WARNING: vLLM did not answer /v1/models within 20 minutes"; fi
    local dur=$(( t_up - T_STOP ))
    printf '%s job=%s stop=%s up=%s down=%d.%d min label="%s" serving=%s\n' "$(date +%F)" "$JOB" \
      "$(date -r "$T_STOP" +%H:%M:%S)" "$(date +%H:%M:%S)" $((dur / 60)) $(( (dur % 60) * 10 / 60 )) "$LABEL" "$ok" >> "$DOWNTIME_LOG"
  fi
  log "done (status $rc)"
  exit $rc
}
trap cleanup EXIT INT TERM

log "window job $JOB: $LABEL"
STOPPED=1; T_STOP=$(date +%s)   # set first: if docker stop half-fails the trap still starts the container
rsh "docker stop $VLLM_CONTAINER >/dev/null"
ok=0
for i in $(seq 1 60); do
  gb=$(mem_avail_gb)
  (( gb >= 70 )) && { log "memory freed: ${gb} GB available"; ok=1; break; }
  sleep 5
done
[[ $ok == 1 ]] || { log "MemAvailable did not recover after stopping vLLM"; exit 1; }

log "running $SCRIPT (cap ${MAXMIN} min)"
RLOG=$(mktemp)
rsh "timeout ${MAXMIN}m bash $SCRIPT" > "$RLOG" 2>&1 &
SPID=$!
done_arms=" "; shown=0
while kill -0 $SPID 2>/dev/null; do
  n=$(wc -l < "$RLOG"); (( n > shown )) && { sed -n "$((shown + 1)),${n}p" "$RLOG"; shown=$n; }
  for arm in ${LOCALEVAL_ARMS:-}; do
    [[ "$done_arms" == *" $arm "* ]] && continue
    grep -q "^READY $arm" "$RLOG" || continue
    log "localeval arm $arm"
    (cd /opt/code/deepeval && FLEETQ_JOB=$JOB uv run localeval speed --url ${LE_URL:-http://spark-fa00:8090/v1} --model intarsia-35b \
      --thinking off --sizes "${LE_SIZES:-1024,4096,16384}" --gen 128 --reps 3 --label "${LE_PREFIX:-intarsia-s1}-$arm") 2>&1 | tail -25
    log "localeval arm $arm exit ${PIPESTATUS[0]}"
    rsh "touch ~/intarsia-decfuse/out/${LE_OUTNAME:-s1W}/$arm.done"
    done_arms="$done_arms$arm "
  done
  sleep 3
done
wait $SPID; rc=$?; SPID=""
n=$(wc -l < "$RLOG"); (( n > shown )) && sed -n "$((shown + 1)),${n}p" "$RLOG"
rm -f "$RLOG"
log "script exit $rc"
exit $rc
