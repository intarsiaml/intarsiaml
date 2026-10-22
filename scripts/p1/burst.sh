#!/usr/bin/env bash
# Lane P1: one GPU burst on tatooine under the radiance downtime rule. Runs ON tatooine inside ONE fleetq run:
#   fleetq run --host tatooine --takes-down radiance --mem 30 --label "p1: ..." -- ssh tatooine 'FLEETQ_JOB=$FLEETQ_JOB bash ~/intarsiaml-p1/scripts/p1/burst.sh <stage>...'
# Stops radiance, runs scripts/p1/remote.sh stages (hard cap BURST_CAP s, default 1140 = 19 min), and on EXIT restarts
# radiance and polls /v1/models (<= 3 min). Logs stop / start / answered timestamps (the downtime log) to ~/p1-logs/.
set -uo pipefail
LOGDIR=$HOME/p1-logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/burst-$(date +%m%d-%H%M%S).log
now() { date '+%F %T'; }
restart() {
  docker rm -f p1-serve >/dev/null 2>&1
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
timeout "${BURST_CAP:-1140}" bash "$HOME/intarsiaml-p1/scripts/p1/remote.sh" "$@" 2>&1 | tee -a "$LOG"
echo "[burst] stages done $(now) (timeout/cap exit ${PIPESTATUS[0]})" | tee -a "$LOG"
