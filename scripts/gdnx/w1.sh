#!/usr/bin/env bash
# Lane gdnx, GPU window 1 (one fleetq slot on spark1, quiet on GPU and CPU): batched == solo on the 35B (batch-real.sh, dev test
# binary, default env), then timing_35b_batched_step under the env variants, two interleaved rounds.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 40 --expect 45 --label "gdnx: W1 batch_cuda_real + step timing" -- scripts/gdnx/w1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=$H/../../docs/results/2026-10-29-gdnx-decode-kernels/raw; mkdir -p $RAW
date +%T; "$H/batch-real.sh" dev > "$RAW/batch-real.txt" 2>&1; cat "$RAW/batch-real.txt"
grep -aq "test result: ok. 2 passed" "$RAW/batch-real.txt" || { echo "ABORT: batch_cuda_real"; exit 3; }
OFF=INTARSIA_GDN_DEC=0,INTARSIA_CUDA_FP8_PAIR_ROWS=0
V="$OFF - INTARSIA_CUDA_FP8_PAIR_ROWS=0 INTARSIA_GDN_DEC=0 INTARSIA_SIDE_PROJ=0 INTARSIA_SIDE_PROJ=0,INTARSIA_CUDA_FP8_PAIR_ROWS=0"
"$H/steptime.sh" $V $(echo $V | tr ' ' '\n' | tail -r | tr '\n' ' ') | tee "$RAW/steptime-w1.txt"
date +%T
