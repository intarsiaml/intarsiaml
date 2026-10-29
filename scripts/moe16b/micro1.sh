#!/usr/bin/env bash
# Lane moe16b, micro job (< 3 min, not a window): fp8 rows exactness incl. the L2-prefetch forms, the evicted micro-timing of the
# prefetch forms (all 35B fp8 shapes, M 2..16), and an ncu summary of the 16-row 2048x4096 and 8192x2048 calls, unset vs pf2.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 10 --expect 6 --label "moe16b: fp8 rows micro" -- scripts/moe16b/micro1.sh <out>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
OUT=/opt/code/intarsiaml/.worktrees/moe16b/docs/results/2026-10-28-moe16b-step-overlap/raw/$1; mkdir -p "$OUT"
rsh() { ssh -n -o BatchMode=yes spark1 "export PATH=/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB INTARSIA_REQUIRE_GPU=1; cd ~/intarsiaml-moe16b/src && T=\$(ls -t target/release/deps/rows_exact-* | grep -v '\.d\$' | head -1); $*"; }
date +%T
rsh 'timeout 100 $T --test-threads=3 fp8_rows 2>&1 | grep -aE "^test |G-B|panicked|FAIL|result" | cut -c1-300' | tee "$OUT/exact.txt"
date +%T
rsh 'MSD2_ONLY=fp8pf timeout 60 $T --ignored --nocapture timing_rows_evict 2>&1 | grep -aE "^EVT|panicked"' | tee "$OUT/evict.txt"
date +%T
for sh in 2048x4096 8192x2048; do
  for f in unset:40 pf2:536; do  # the forms run in order, 8 + 240 launches each: unset, pf1, pf2, ...
    pf=${f%%:*}; skip=${f##*:}
    rsh "MSD2_ONLY=fp8pf MSD2_SHAPE=$sh MSD2_M=16 timeout 60 ncu --kernel-name regex:fp8_gemv_rows_q --launch-skip $skip --launch-count 1 --section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats \$T --ignored --nocapture timing_rows_evict 2>&1 | grep -aE 'Duration|Throughput|Occupancy|Registers|Block Limit|Warp Cycles|Stall|Issued|Eligible|Active Warps|No Eligible|One or More|Grid Size|Block Size|Waves' | cut -c1-160" > "$OUT/ncu-$sh-$pf.txt"
    echo "ncu $sh $pf: $(grep -c . "$OUT/ncu-$sh-$pf.txt") lines"
  done
done
date +%T
