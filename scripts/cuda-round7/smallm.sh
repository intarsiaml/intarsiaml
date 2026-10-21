#!/usr/bin/env bash
# Round 7: the e4m3 GEMMs at batched-decode sizes (M = 8, 32, 128), vLLM up (relative numbers), no rebuild:
#   fleetq run --host spark1 --mem 8 --label "..." -- scripts/cuda-round7/smallm.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for m in 8 32 128; do
  NOBUILD=1 STEPS=wq WQ_M=$m WQ_KERN=matmul_fp8_gemm_reg,matmul_fp8_gemm_lt "$S/dev.sh" | grep -E "M=|panicked"
done
