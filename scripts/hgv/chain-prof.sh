#!/usr/bin/env bash
# Lane hgv: the gates, then one rocprofv3 window per (arm, tree), each its own fleetq job (run FROM THE MAC in the worktree).
set -u
R=$PWD/docs/results/2026-10-30-hgv-hip-small-gemv-grid/raw
H="--host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --mem 30"
STEPS="gates" fleetq run $H --expect 12 --label "HGV gates (final kernels)" -- scripts/hgv/gate.sh > $R/fleetq-gate-final.log 2>&1
for cfg in tp1:hgv tp1:hgvbase tp2:hgvbase tp2:hgv; do
  arm=${cfg%%:*}; tree=${cfg##*:}
  ARM=$arm STEPS=prof TREE=$tree PHASES="same16,div16" CAP=900 OUT=$R/prof-$arm-$tree-final \
    fleetq run $H --expect 12 --label "HGV prof $arm $tree" -- scripts/hprof/window.sh > $R/fleetq-prof-$arm-$tree-final.log 2>&1
done
echo chain done
