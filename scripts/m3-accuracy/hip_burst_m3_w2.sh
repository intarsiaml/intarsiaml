#!/usr/bin/env bash
# ON tatooine: one radiance-down window for M3 (pattern of scripts/hip-decode/burst.sh). trap restores radiance on any exit.
set -uo pipefail
LOG=$HOME/hipburst-logs/m3-$(date +%m%d-%H%M%S).log; mkdir -p $HOME/hipburst-logs
now() { date '+%F %T'; }
restart() {
  docker rm -f intarsia-serve >/dev/null 2>&1
  echo "[burst] starting radiance $(now)" | tee -a "$LOG"
  systemctl --user start radiance
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then
      echo "[burst] radiance answers $(now) (after $((i * 5)) s)" | tee -a "$LOG"; return 0
    fi
    sleep 5
  done
  echo "[burst] RADIANCE DID NOT ANSWER within 180 s $(now)" | tee -a "$LOG"; return 1
}
trap restart EXIT
echo "[burst] job ${FLEETQ_JOB:-none}; M3 window; stopping radiance $(now)" | tee "$LOG"
systemctl --user stop radiance
sleep 2
echo "[burst] radiance stopped $(now)" | tee -a "$LOG"
timeout "${BURST_CAP:-420}" bash $HOME/intarsiaml-m3/m3data/hip_w2_remote.sh 2>&1 | tee -a "$LOG"
echo "[burst] stages done $(now) (exit ${PIPESTATUS[0]})" | tee -a "$LOG"
