#!/usr/bin/env bash
# One GPU window of lane ink-hip-batch. Runs ON tatooine, inside a fleetq job that takes radiance down:
#   fleetq run --host tatooine-r9700 --takes-down radiance --expect <MIN> --label "ink-hip-attn: <what>" -- \
#     ssh -o BatchMode=yes tatooine 'CAP=<seconds> bash ~/intarsiaml-ink-hip-batch/.scratch/window.sh <stage script> [args]' < /dev/null
# Stops radiance, runs the stage script in its own process group under a hard cap (CAP seconds, default 1500 = 25 min; never
# above 2400), then restarts radiance on ANY exit (trap) and polls /v1/models (<= 3 min). Kills only what it started: the
# stage's process group and the container `ink-hip-batch-run`. Log (the downtime record): ~/intarsiaml-ink-hip-batch/logs/window-*.log.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-batch
mkdir -p "$REPO/logs"
LOG=$REPO/logs/window-$(date +%m%d-%H%M%S).log
CAP=${CAP:-1500}
[ "$CAP" -gt 2400 ] && CAP=2400
now() { date '+%F %T'; }
STAGE_PID=
restored=0
restore() {
  [ "$restored" = 1 ] && return 0
  restored=1
  if [ -n "$STAGE_PID" ] && kill -0 "$STAGE_PID" 2>/dev/null; then
    kill -TERM -- "-$STAGE_PID" 2>/dev/null
    sleep 2
    kill -KILL -- "-$STAGE_PID" 2>/dev/null
  fi
  docker rm -f ink-hip-batch-run >/dev/null 2>&1
  echo "[window] starting radiance $(now)" | tee -a "$LOG"
  systemctl --user start radiance
  local i
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then
      echo "[window] radiance answers $(now) (after $((i * 5)) s)" | tee -a "$LOG"
      return 0
    fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 180 s $(now)" | tee -a "$LOG"
  return 1
}
trap restore EXIT
trap 'echo "[window] signal $(now)" | tee -a "$LOG"; exit 143' INT TERM HUP
echo "[window] job ${FLEETQ_JOB:-none}; stage: $*; cap ${CAP}s; stopping radiance $(now)" | tee "$LOG"
systemctl --user stop radiance
sleep 3
echo "[window] radiance stopped $(now) ($(systemctl --user is-active radiance))" | tee -a "$LOG"
setsid timeout "$CAP" bash "$@" < /dev/null >> "$LOG" 2>&1 &
STAGE_PID=$!
tail -n 0 -f "$LOG" --pid "$STAGE_PID" 2>/dev/null &
wait "$STAGE_PID"
rc=$?
sleep 1
echo "[window] stage done $(now) (exit $rc; 124 = cap hit)" | tee -a "$LOG"
restore
r=$?
[ "$rc" -ne 0 ] && exit "$rc"
exit "$r"
