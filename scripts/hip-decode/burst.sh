#!/usr/bin/env bash
# One GPU burst on tatooine with the radiance downtime rule. Runs ON tatooine, inside a fleetq lease the caller took
# (`fleetq acquire --host tatooine --takes-down radiance --mem 30 --label ... --wait`, then
# `FLEETQ_JOB=<id> ssh tatooine 'bash ~/intarsiaml-hipperf/scripts/hip-decode/burst.sh <stage>...'`, then `fleetq release <id>`
# only after this script printed "radiance answers"). Stops radiance, runs the stages of remote.sh (hard cap BURST_CAP seconds, default 900 = 15 min),
# restarts radiance and polls /v1/models (<= 3 min). Logs with the stop / start / answered timestamps (the downtime log) to
# ~/hipburst-logs/.
set -uo pipefail
LOGDIR=$HOME/hipburst-logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/burst-$(date +%m%d-%H%M%S).log
now() { date '+%F %T'; }
restart() {
  docker rm -f intarsia-serve >/dev/null 2>&1
  echo "[burst] starting radiance $(now)" | tee -a "$LOG"
  systemctl --user start radiance
  local i
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then
      echo "[burst] radiance answers $(now) (after $((i * 5)) s)" | tee -a "$LOG"
      return 0
    fi
    sleep 5
  done
  echo "[burst] RADIANCE DID NOT ANSWER within 180 s $(now)" | tee -a "$LOG"
  return 1
}
trap restart EXIT
echo "[burst] job ${FLEETQ_JOB:-none}; stages: $*; stopping radiance $(now)" | tee "$LOG"
systemctl --user stop radiance
sleep 2
echo "[burst] radiance stopped $(now)" | tee -a "$LOG"
timeout "${BURST_CAP:-900}" bash "${INTARSIA_REPO:-$HOME/intarsiaml-hipperf}/scripts/hip-decode/remote.sh" "$@" 2>&1 | tee -a "$LOG"
echo "[burst] stages done $(now) (timeout/cap exit ${PIPESTATUS[0]})" | tee -a "$LOG"
