#!/usr/bin/env zsh
# fx6: a third sameslot run (extra visits for the noisy Metal prefill cells), same binary as metal-job.sh, order rotated again.
# fleetq run --host endor --exclusive --mem 24 --expect 20 --label "fx6 metal run3" -- zsh scripts/standings-fx6/metal-run3.sh <out dir>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside fleetq slot" >&2; exit 2; }
OUT=${1:?out dir}
cd ${0:A:h:h:h}
BIN=$PWD/target-metal/release/intarsia
[ -x $BIN ] || { echo "no binary" >&2; exit 3; }
echo "run 3 start $(date)" | tee -a $OUT/job.log
ORDER="mlx-lm intarsia-1 llamacpp intarsia-2 mlx-lm-2" zsh scripts/metal-speed-gate/sameslot.sh $BIN $OUT/run3 2>&1 | tee -a $OUT/job.log
echo "done run 3 $(date)" | tee -a $OUT/job.log
