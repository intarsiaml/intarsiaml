#!/usr/bin/env bash
# Lane moedn, GPU window 1 (one fleetq slot on spark1, quiet on GPU and CPU):
#   1. exactness gates (gates.sh: quantmoe, moe_nvfp4, rows_exact, row_tables, prefix_store_restore_is_bitwise_cold_cuda);
#   2. batch_cuda_real 35B (batched == solo, dev binary, default env = dedup down kernel);
#   3. timing_35b_batched_step B = 8 / 16, one dev binary, INTARSIA_MOE_DOWN_DD=0 (v2 down) vs default, 2 interleaved rounds;
#   4. nsys per-kernel buckets at B = 16 (serve burst, VPROF's units): dev default, then dev with INTARSIA_MOE_DOWN_DD=0.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 45 --expect 45 --label "moedn: W1 gates + step + nsys" -- scripts/moedn/w1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=$H/../../docs/results/2026-10-29-moedn-down-proj/raw; mkdir -p $RAW
T0=$SECONDS
if [[ -z "${SKIP_GATES:-}" ]]; then
  date +%T; "$H/gates.sh" 2>&1 | tee "$RAW/gates.txt"
  grep -aq "test result: ok" "$RAW/gates-rows_exact.txt" && ! grep -aq "FAILED\|panicked" "$RAW/gates-rows_exact.txt" || { echo "ABORT: rows_exact"; exit 3; }
fi
date +%T; "$H/batch-real.sh" dev > "$RAW/batch-real.txt" 2>&1; cat "$RAW/batch-real.txt"
grep -aq "test result: ok. 2 passed" "$RAW/batch-real.txt" || { echo "ABORT: batch_cuda_real"; exit 3; }
OFF=INTARSIA_MOE_DOWN_DD=0
"$H/steptime.sh" $OFF - - $OFF | tee "$RAW/steptime-w1.txt"
date +%T; "$H/prof.sh" dev dev-b16 2>&1 | tail -3
if (( SECONDS - T0 < 31*60 )); then date +%T; "$H/prof.sh" dev dev-off-b16 $OFF 2>&1 | tail -3; else echo "SKIP prof dev-off (window time)"; fi
date +%T
