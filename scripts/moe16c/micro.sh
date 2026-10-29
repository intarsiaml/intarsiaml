#!/usr/bin/env bash
# Lane moe16c micro (< 3 min, fleetq --quiet gpu,cpu): rows_exact (every form, M = 1..16) on the dev binary, then the evicted
# row-kernel timing (fp8 and nvfp4 forms) of base and dev, interleaved base / dev / dev / base. Prints `EVT ...` lines.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
L=/home/nvidia/intarsiaml-moe16c/bin
ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-moe16c/src/crates/intarsia-backend-cuda; date +%T;
  [ -n \"${SKIP_EXACT:-}\" ] || INTARSIA_REQUIRE_GPU=1 timeout 80 $L/rows_exact-dev --test-threads=1 2>&1 | grep -aE '^test |test result|panicked|mismatch' | head -40; date +%T;
  for v in ${VARIANTS:-base dev dev base}; do echo \"== \$v\"; MSD2_ONLY=${ONLY:-fp8,nvfp4} timeout 40 $L/rows_exact-\$v --ignored --nocapture --test-threads=1 timing_rows_evict 2>&1 | grep -aE '^EVT|panicked|test result'; date +%T; done"
