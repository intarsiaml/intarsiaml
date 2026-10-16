#!/usr/bin/env bash
# One short burst of GPU work on tatooine's two R9700s with the radiance downtime rule:
#   fleetq run (lease: both cards, takes radiance down) -> stop radiance -> run stages -> start radiance ->
#   poll until it answers (<= 3 min) -> the lease is released by fleetq run when this script exits.
#   scripts/phase2b-burst.sh "<label>" stage [stage...]       (stages: scripts/phase2b-tp-remote.sh)
# Logs to $BURST_LOG_DIR (default /tmp) with the stop / start / answered timestamps (the downtime log).
set -uo pipefail
if [ -z "${FLEETQ_JOB:-}" ]; then
  label=$1
  exec fleetq run --host tatooine-r9700,tatooine-r9700-2 --mem 30 --takes-down radiance --label "intarsiaml tp2: $label" -- "$0" "$@"
fi
shift
LOGDIR=${BURST_LOG_DIR:-/tmp}
mkdir -p "$LOGDIR"
LOG=$LOGDIR/burst-$(date +%H%M%S).log
now() { date '+%F %T'; }
restart() {
  echo "[burst] starting radiance $(now)" | tee -a "$LOG"
  ssh tatooine 'systemctl --user start radiance' | tee -a "$LOG"
  for i in $(seq 1 36); do
    if curl -s -m 3 http://tatooine:8080/v1/models | grep -q .; then
      echo "[burst] radiance answers $(now) (after $((i * 5)) s)" | tee -a "$LOG"
      return 0
    fi
    sleep 5
  done
  echo "[burst] RADIANCE DID NOT ANSWER within 180 s $(now)" | tee -a "$LOG"
  return 1
}
trap restart EXIT
echo "[burst] job ${FLEETQ_JOB}; stopping radiance $(now)" | tee "$LOG"
ssh tatooine 'systemctl --user stop radiance; sleep 3; systemctl --user is-active radiance; true' | tee -a "$LOG"
echo "[burst] radiance stopped $(now)" | tee -a "$LOG"
ssh tatooine "INTARSIA_REPO=\$HOME/intarsiaml-tp bash \$HOME/intarsiaml-tp/scripts/phase2b-tp-remote.sh $*" 2>&1 | tee -a "$LOG"
echo "[burst] stages done $(now)" | tee -a "$LOG"
