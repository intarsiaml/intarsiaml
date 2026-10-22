#!/usr/bin/env bash
# One radiance-down window on tatooine, same rule as scripts/hip-decode/burst.sh: stop radiance, run ONE command under a hard
# cap, ALWAYS (trap) remove our containers, start radiance and poll /v1/models. Runs ON tatooine inside a fleetq job that
# takes radiance down (the caller releases that job only after "radiance answers" was printed).
#   og-burst.sh <cap-seconds> <label> <command...>
set -uo pipefail
CAP=${1:?cap seconds}; LABEL=${2:?label}; shift 2
LOGDIR=$HOME/hipburst-logs; mkdir -p "$LOGDIR"
LOG=$LOGDIR/og-$(date +%m%d-%H%M%S).log
now() { date '+%F %T'; }
# at least 5 min of serving since the previous window (any lane that logs to ~/hipburst-logs)
last=$(ls -t "$LOGDIR"/*.log 2>/dev/null | head -1)
if [[ -n "$last" ]]; then age=$(( $(date +%s) - $(stat -c %Y "$last") )); (( age < 300 )) && { echo "[og-burst] previous window ended ${age}s ago; waiting $((300 - age))s so radiance serves 5 min"; sleep $((300 - age)); }; fi
T0=$(date +%s)
restart() {
  docker rm -f og-serve >/dev/null 2>&1
  # any container still mounting OUR cargo target volume (a dev.sh run the timeout orphaned); other lanes use other volumes
  docker ps -q --filter volume=${INTARSIA_TARGET_VOL:-intarsia-cargo-target-og} | xargs -r docker kill >/dev/null 2>&1
  echo "[og-burst] starting radiance $(now)" | tee -a "$LOG"
  systemctl --user start radiance
  local i
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then
      echo "[og-burst] radiance answers $(now) (after $((i * 5)) s); window total $(( $(date +%s) - T0 )) s" | tee -a "$LOG"
      return 0
    fi
    sleep 5
  done
  echo "[og-burst] RADIANCE DID NOT ANSWER within 180 s $(now)" | tee -a "$LOG"
  return 1
}
trap restart EXIT
echo "[og-burst] job ${FLEETQ_JOB:-none}; $LABEL; cap ${CAP}s; stopping radiance $(now)" | tee "$LOG"
systemctl --user stop radiance
sleep 2
echo "[og-burst] radiance stopped $(now)" | tee -a "$LOG"
timeout "$CAP" "$@" 2>&1 | tee -a "$LOG"
echo "[og-burst] command done $(now) (exit ${PIPESTATUS[0]})" | tee -a "$LOG"
