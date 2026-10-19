#!/usr/bin/env bash
# One radiance-down GPU window for core-op fusion phase 1 (docs/results/2026-10-19-core-op-fusion-p1.md); burst.sh with two
# lanes. Runs ON tatooine inside a lease the caller took:
#   fleetq acquire --host tatooine --takes-down radiance --mem 30 --label "..." --wait
#   FLEETQ_JOB=<id> ssh tatooine 'FLEETQ_JOB=<id> bash ~/intarsiaml-fusion/scripts/hip-decode/fusion-window.sh "<gpu0 stages>" "<gpu1 stages>"'
#   fleetq release <id>      (only after this printed "radiance answers")
# Both lanes run fusion-p1.sh stages (FUSION_SCRIPT=fusion-p2.sh: phase 2's), GPU 0 (MUT_GPU=TEST_GPU=0) and GPU 1 (=1) in parallel; on GPU 0 a stage "r:<stage>" is
# remote.sh's (A/B against BASE_REPO with its @base suffix; remote.sh pins HIP_VISIBLE_DEVICES=0). Hard cap BURST_CAP seconds (default 900) on both lanes; radiance is restarted on exit (trap) and
# polled on /v1/models. Downtime lines in ~/fusion-logs/.
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-fusion}" INTARSIA_TARGET_VOL="${INTARSIA_TARGET_VOL:-intarsia-cargo-target-fusion}"
export BASE_REPO="${BASE_REPO:-$HOME/intarsiaml-fusionbase}" BASE_VOL="${BASE_VOL:-intarsia-cargo-target-fusionbase}"
LOGDIR=$HOME/fusion-logs
mkdir -p "$LOGDIR"
TAG=$(date +%m%d-%H%M%S)
LOG=$LOGDIR/window-$TAG.log
now() { date '+%F %T'; }
restart() {
  wait 2>/dev/null
  docker rm -f intarsia-serve >/dev/null 2>&1
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
trap restart EXIT
echo "[window] job ${FLEETQ_JOB:-none}; gpu0: $1; gpu1: ${2:-}; stopping radiance $(now)" | tee "$LOG"
systemctl --user stop radiance
sleep 2
echo "[window] radiance stopped $(now)" | tee -a "$LOG"
if [ -n "${2:-}" ]; then
  ( MUT_GPU=1 TEST_GPU=1 timeout "${BURST_CAP:-900}" bash "$INTARSIA_REPO/scripts/hip-decode/${FUSION_SCRIPT:-fusion-p1.sh}" $2 > "$LOGDIR/window-$TAG-gpu1.log" 2>&1
    echo "[window] gpu1 lane done $(now) (exit $?)" >> "$LOG" ) &
fi
MUT_GPU=0 TEST_GPU=0 timeout "${BURST_CAP:-900}" bash "$INTARSIA_REPO/scripts/hip-decode/${FUSION_SCRIPT:-fusion-p1.sh}" $1 2>&1 | tee -a "$LOG"
echo "[window] gpu0 lane done $(now) (exit ${PIPESTATUS[0]})" | tee -a "$LOG"
wait
cat "$LOGDIR/window-$TAG-gpu1.log" 2>/dev/null | sed 's/^/[gpu1] /' >> "$LOG"
