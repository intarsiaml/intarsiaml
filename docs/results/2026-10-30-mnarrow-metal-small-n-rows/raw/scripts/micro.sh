#!/bin/bash
# MNARROW microbench + exactness: usage micro.sh OUTDIR TAG (inside a fleetq slot). rows_exact (all gates) once, then timing_rows
# 3x per arm (arms: default, INTARSIA_METAL_NV_ROWS_ONE=0 = two chunks), alternating.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
O=$1; TAG=$2; B=/opt/code/intarsiaml/.worktrees/mnarrow/.lane/bins/$TAG/rows_exact; mkdir -p $O
export INTARSIA_REQUIRE_GPU=1
[ "${SKIPGATE:-0}" = 1 ] || { $B --test-threads=1 --nocapture > $O/rows_exact.log 2>&1; echo "rows_exact exit $?"; grep -E "test result|^test " $O/rows_exact.log | cut -c1-160; }
for r in 1 2 3; do
  for arm in "X=1" "INTARSIA_METAL_NV_ROWS_ONE=0"; do
    env $arm $B --ignored --test-threads=1 --nocapture timing_rows > $O/timing-$(echo $arm | tr -c 'A-Za-z0-9\n' _)-r$r.log 2>&1; echo "timing $arm r$r exit $?"
  done
done
