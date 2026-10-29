#!/usr/bin/env bash
# Lane stream16, GPU window 2 (one fleetq slot on spark1, quiet on GPU and CPU):
#   1. probe round 2 (prologue overlap, no-quant diagnostic, lm_head shapes at M = 16 / 12 / 8);
#   2. gates: rows_exact (streaming tests + the rest), row_tables, batch_cuda_real 35B pairs (dev, defaults);
#   3. dev micro-timing (evicted fp8 + lm_head), default and with both knobs off;
#   4. step A/B base / dev interleaved at B = 1, 4, 8, 16 (base twice and dev twice: identical-arm noise), dev knobs-off.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 45 --expect 40 --label "stream16: W2 probe2 + gates + step A/B" -- scripts/stream16/w2.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; RAW=$H/../../docs/results/2026-10-29-stream16-gemv-stream/raw; mkdir -p $RAW
L='~/intarsiaml-stream16'; T0=$SECONDS
rsh() { ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
OFF="INTARSIA_CUDA_FP8_ROWS_QPIPE=0+INTARSIA_CUDA_NVFP4_ROWS_PIPE=0"
date +%T; rsh "cd $L/probe && timeout 200 ./probe16 all; timeout 60 ./probe12 nv; timeout 60 ./probe8 nv" 2>&1 | tee $RAW/probe-w2.txt
date +%T; "$H/gates.sh" 2>&1 | tee $RAW/gates.txt | grep -aE "^==|test result|FAIL|panicked"
date +%T; "$H/batch-real.sh" dev 2>&1 | tee $RAW/batch-real.txt | grep -aE "test result|FAIL|panicked"
date +%T; rsh "cd $L/src/crates/intarsia-backend-cuda && INTARSIA_REQUIRE_GPU=1 MSD2_ONLY=fp8nvfp4 timeout 200 $L/bin/rows_exact-dev --ignored --nocapture --test-threads=1 --exact timing_rows_evict 2>&1 | grep -aE '^EVT.*M=(1|8|12|16) |panicked'" | tee $RAW/micro-evict-dev.txt
rsh "cd $L/src/crates/intarsia-backend-cuda && INTARSIA_CUDA_FP8_ROWS_QPIPE=0 INTARSIA_REQUIRE_GPU=1 MSD2_ONLY=fp8nvfp4 timeout 200 $L/bin/rows_exact-dev --ignored --nocapture --test-threads=1 --exact timing_rows_evict 2>&1 | grep -aE '^EVT.*M=(1|8|12|16) |panicked'" | tee $RAW/micro-evict-devoff.txt
date +%T; SB=1,4,8,16 "$H/steptime.sh" base@- dev@- base@- dev@- dev@$OFF dev@- base@- 2>&1 | tee $RAW/steptime-w2.txt
echo "window used $(( (SECONDS - T0) / 60 )) min"; date +%T
