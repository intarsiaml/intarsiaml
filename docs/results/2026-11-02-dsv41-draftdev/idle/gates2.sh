#!/bin/bash
# Under fleetq, spark1: the 3-layer bitwise test on CUDA, eager and replayed. (Engram off in this test, so the Engram switch is a no-op here;
# the TP=2 arms cover it.)
set -u
: "${FLEETQ_JOB:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"; R="$HERE/../../scripts/dsv41-tp2"
C='INTARSIA_DSV41_BACKEND=cuda INTARSIA_DSV41_CACHE=$HOME/intarsia-dsv41-tp2/cache/l3-r0of1 INTARSIA_DSV41_DP_SPEC=1 INTARSIA_DSV41_DPOS_GRAPHS=1'
rc=0
for i in 1 2; do
  "$R/spark-test.sh" spark1 dsv41_runtime device_position_passes_equal_host_passes $C || rc=1
  "$R/spark-test.sh" spark1 dsv41_runtime device_position_passes_equal_host_passes $C INTARSIA_DSV41_ENGRAM_ONEGRAPH=1 || rc=1
done
exit $rc
