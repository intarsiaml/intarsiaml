#!/usr/bin/env bash
# One tatooine radiance-down window for a stage of this lane, as ONE fleetq run (radiance back before the slot is released):
#   fleetq run --host tatooine --takes-down radiance --mem 30 --expect <min> --label "og: ..." -- scripts/og-gates/hip-window.sh o5|o4 <cap-seconds>
# Syncs the scripts, runs og-burst.sh (stop radiance, command under the cap, trap restore + wait for /v1/models) and brings the
# outputs back to docs/results/raw/2026-10-22-og-gates/. The downtime goes to downtime.log.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run --host tatooine --takes-down radiance}"
STAGE=${1:?o5|o4}; CAP=${2:?cap seconds}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW=$REPO/docs/results/raw/2026-10-22-og-gates; mkdir -p "$RAW"
HOST=tatooine
rsh() { ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsync -a "$REPO/scripts/og-gates/" "$HOST:og-hip/scripts/og-gates/"
rsh "cd ~/og-hip && INTARSIA_REPO=\$HOME/og-hip INTARSIA_TARGET_VOL=intarsia-cargo-target-og deploy/tatooine/dev.sh bash -c 'test -x /target/release/intarsia && /target/release/intarsia --version'" || { echo "HIP binary missing on tatooine: run hip-build-tatooine.sh first (radiance untouched)"; exit 3; }
case $STAGE in
  o5) CMD="${PRE:+env PRE='$PRE'} bash \$HOME/og-hip/scripts/og-gates/o5-tails.sh hip \$HOME/og-hip/og-out/o5-hip ${WHICH:-all}"; BACK=o5-hip ;;
  o4) CMD="bash \$HOME/og-hip/scripts/og-gates/o4-ab.sh \$HOME/og-hip/og-out/o4 ${ROUNDS:-5}"; BACK=o4 ;;
  *) echo "stage?"; exit 2 ;;
esac
T0=$(date +%s); S0=$(date +%T)
rsh "bash \$HOME/og-hip/scripts/og-gates/og-burst.sh $CAP 'og $STAGE' $CMD" 2>&1 | tee "$RAW/window-$STAGE.log"
T1=$(date +%s)
rsync -a "$HOST:og-hip/og-out/$BACK/" "$RAW/$BACK/"
printf '%s job=%s stage=%s start=%s end=%s wall=%d s (radiance down for the burst; see window-%s.log for stop/answers times)\n' "$(date +%F)" "$FLEETQ_JOB" "$STAGE" "$S0" "$(date +%T)" $((T1 - T0)) "$STAGE" >> "$RAW/downtime.log"
