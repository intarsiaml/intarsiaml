#!/usr/bin/env bash
# Lane qkv16, GPU window 1 (one fleetq slot on spark1, quiet on GPU and CPU):
#   1. exactness gates (gates.sh: quantmoe, fused_decode, rows_exact, row_tables);
#   2. batch_cuda_real 35B (batched == solo, dev binary, default env: the triple from 12 rows), then the batched-decode test again with
#      INTARSIA_CUDA_FP8_TRIPLE_ROWS=2 (the triple at every B: 2, 3, 4, 8, 16);
#   3. timing_35b_batched_step B = 8 / 16, one dev binary, TRIPLE_ROWS=0 (q + k/v pair) vs default vs TRIPLE_ROWS=2, interleaved twice;
#   4. nsys per-kernel buckets at B = 16 (serve burst, VPROF's units): dev default, then dev with TRIPLE_ROWS=0.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 45 --expect 40 --label "qkv16: W1 gates + step + nsys" -- scripts/qkv16/w1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=$H/../../docs/results/2026-10-29-qkv16-triple-gemv/raw; mkdir -p $RAW
T0=$SECONDS
if [[ -z "${SKIP_GATES:-}" ]]; then
  date +%T; "$H/gates.sh" 2>&1 | tee "$RAW/gates.txt"
  grep -aq "test result: ok" "$RAW/gates-rows_exact.txt" && ! grep -aq "FAILED\|panicked" "$RAW/gates-rows_exact.txt" || { echo "ABORT: rows_exact"; exit 3; }
fi
date +%T; "$H/batch-real.sh" dev > "$RAW/batch-real.txt" 2>&1; cat "$RAW/batch-real.txt"
grep -aq "test result: ok. 2 passed" "$RAW/batch-real.txt" || { echo "ABORT: batch_cuda_real"; exit 3; }
date +%T; ENVS=INTARSIA_CUDA_FP8_TRIPLE_ROWS=2 TEST=real_35b_batched_decode "$H/batch-real.sh" dev > "$RAW/batch-real-t2.txt" 2>&1; cat "$RAW/batch-real-t2.txt"
grep -aq "test result: ok. 1 passed" "$RAW/batch-real-t2.txt" || { echo "ABORT: batch_cuda_real TRIPLE_ROWS=2"; exit 3; }
OFF=INTARSIA_CUDA_FP8_TRIPLE_ROWS=0; T2=INTARSIA_CUDA_FP8_TRIPLE_ROWS=2
"$H/steptime.sh" $OFF - $T2 $T2 - $OFF | tee "$RAW/steptime-w1.txt"
date +%T; "$H/prof.sh" dev dev-b16 2>&1 | tail -3
if (( SECONDS - T0 < 28*60 )); then date +%T; "$H/prof.sh" dev dev-off-b16 $OFF 2>&1 | tail -3; else echo "SKIP prof dev-off (window time)"; fi
date +%T
