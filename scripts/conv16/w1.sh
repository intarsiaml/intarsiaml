#!/usr/bin/env bash
# Lane conv16, GPU window 1 (one fleetq slot on spark1, quiet on GPU and CPU):
#   1. exactness gates (gates.sh: gdn_conv16 first, rows_exact, row_tables, prefix_store_restore_is_bitwise_cold_cuda);
#   2. batch_cuda_real 35B (batched == solo, dev binary, default env = bf16 conv state);
#   3. timing_35b_batched_step B = 8 / 16, one dev binary, INTARSIA_CUDA_CONV_BF16=0 (f32) vs default (bf16), 2 interleaved rounds;
#   4. nsys per-kernel buckets at B = 16 (dev default, then dev with INTARSIA_CUDA_CONV_BF16=0 if the window has time left).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 45 --expect 45 --label "conv16: W1 gates + step + nsys" -- scripts/conv16/w1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=$H/../../docs/results/2026-10-29-conv16-bf16-conv-state/raw; mkdir -p $RAW
T0=$SECONDS
if [[ -z "${SKIP_GATES:-}" ]]; then
  date +%T; "$H/gates.sh" 2>&1 | tee "$RAW/gates.txt"
  grep -aq "test result: ok. 2 passed" "$RAW/gates-c16.txt" || { echo "ABORT: gdn_conv16"; exit 3; }
fi
date +%T; "$H/batch-real.sh" dev > "$RAW/batch-real.txt" 2>&1; cat "$RAW/batch-real.txt"
grep -aq "test result: ok. 2 passed" "$RAW/batch-real.txt" || { echo "ABORT: batch_cuda_real"; exit 3; }
OFF=INTARSIA_CUDA_CONV_BF16=0
"$H/steptime.sh" $OFF - - $OFF | tee "$RAW/steptime-w1.txt"
date +%T; "$H/prof.sh" dev dev-b16 2>&1 | tail -3
if (( SECONDS - T0 < 33*60 )); then date +%T; "$H/prof.sh" dev dev-off-b16 $OFF 2>&1 | tail -3; else echo "SKIP prof dev-off (window time)"; fi
date +%T
