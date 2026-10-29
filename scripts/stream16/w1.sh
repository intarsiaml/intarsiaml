#!/usr/bin/env bash
# Lane stream16, GPU window 1 (one fleetq slot on spark1, quiet on GPU and CPU):
#   1. the streaming probe (probe.cu: baseline copies of the 16-row fp8 rows_q and lm_head kernels vs load-pipelining variants);
#   2. main's evicted micro-timing of the row-exact GEMVs (rows_exact timing_rows_evict, timing_gemv_rows) -> per-kernel GB/s;
#   3. nsys of one served B = 16 burst and one B = 1 stream on main (prof.sh) -> per-kernel table of the step.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 45 --expect 40 --label "stream16: W1 probe + micro + nsys B16/B1" -- scripts/stream16/w1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; RAW=$H/../../docs/results/2026-10-29-stream16-gemv-stream/raw; mkdir -p $RAW
L='~/intarsiaml-stream16'; T0=$SECONDS
rsh() { ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
date +%T; rsh "cd $L/probe && timeout 230 ./probe all" 2>&1 | tee $RAW/probe-w1.txt
date +%T; rsh "cd $L/src/crates/intarsia-backend-cuda && INTARSIA_REQUIRE_GPU=1 timeout 230 $L/bin/rows_exact-base --ignored --nocapture --test-threads=1 --exact timing_rows_evict 2>&1 | grep -aE '^EVT|panicked|test result'" | tee $RAW/micro-evict-base.txt
date +%T; rsh "cd $L/src/crates/intarsia-backend-cuda && INTARSIA_REQUIRE_GPU=1 timeout 230 $L/bin/rows_exact-base --ignored --nocapture --test-threads=1 --exact timing_gemv_rows 2>&1 | grep -aE '^TIME|panicked|test result'" | tee $RAW/micro-gemv-base.txt
date +%T; "$H/prof.sh" base base-b16 2>&1 | tail -3
if (( SECONDS - T0 < 28*60 )); then date +%T; N=1 "$H/prof.sh" base base-b1 2>&1 | tail -3; else echo "SKIP prof base-b1 (window time)"; fi
date +%T
