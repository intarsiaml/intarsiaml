#!/usr/bin/env bash
# One tatooine radiance-down window for a stage of lane ck, as ONE fleetq run (radiance back before the slot is released):
#   fleetq run --host tatooine --takes-down radiance --mem 30 --expect <min> --label "ck: ..." -- scripts/ck/hip-window.sh acc|dec|speed35|speed8 <cap-seconds>
# Syncs the scripts, runs hip-stage.sh under og-burst.sh (stop radiance, cap, trap restore + wait for /v1/models), brings the
# outputs back to docs/results/2026-10-22-ck-prefill-chunk/raw/hip/ and logs the downtime.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run --host tatooine --takes-down radiance}"
STAGE=${1:?acc|dec|speed35|speed8}; CAP=${2:?cap seconds}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-22-ck-prefill-chunk/raw/hip; mkdir -p "$RAW"
HOST=tatooine
rsh() { ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsync -a "$REPO/scripts/ck/" "$HOST:ck-hip/scripts/ck/"
rsync -a "$REPO/scripts/og-gates/og-burst.sh" "$HOST:ck-hip/scripts/og-gates/"
rsh "cd ~/ck-hip && INTARSIA_REPO=\$HOME/ck-hip INTARSIA_TARGET_VOL=intarsia-cargo-target-ck deploy/tatooine/dev.sh bash -c 'test -x /target/release/intarsia && /target/release/intarsia --version'" || { echo "HIP binary missing on tatooine: run scripts/ck/hip-build-tatooine.sh first (radiance untouched)"; exit 3; }
OUTR=ck-out/$STAGE-$(date +%H%M%S)
T0=$(date +%s); S0=$(date +%T)
rsh "export INTARSIA_REPO=\$HOME/ck-hip INTARSIA_TARGET_VOL=intarsia-cargo-target-ck; bash \$HOME/ck-hip/scripts/og-gates/og-burst.sh $CAP 'ck $STAGE' bash \$HOME/ck-hip/scripts/ck/hip-stage.sh $STAGE \$HOME/ck-hip/$OUTR" 2>&1 | tee "$RAW/window-$STAGE.log"
rsync -a "$HOST:ck-hip/$OUTR/" "$RAW/$STAGE/"
printf '%s job=%s stage=%s start=%s end=%s wall=%d s (radiance down for the burst; see window-%s.log for stop/answers times)\n' "$(date +%F)" "$FLEETQ_JOB" "$STAGE" "$S0" "$(date +%T)" $(( $(date +%s) - T0 )) "$STAGE" >> "$RAW/downtime.log"
