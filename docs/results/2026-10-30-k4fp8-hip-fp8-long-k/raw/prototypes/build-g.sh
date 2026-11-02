#!/usr/bin/env bash
# Run on the GPU box: compile the grouped inner loop (tree kernels, -DK4_G=<rows per group>) and the asm of G=4.
set -uo pipefail
T=$HOME/intarsiaml-tp2-k4fp8; cd "$T"
for g in ${GS:-1 2 4 8}; do bash scripts/k4fp8/bench.sh "$T" ${PFX:-g}$g -DK4_G=$g & done
TAG=${PFX:-g}4 bash scratch/isa.sh -DK4_G=4 &
wait
for g in ${GS:-1 2 4 8}; do
  echo "== g$g"; grep -A9 "Function Name: .*fp8_gemv_rows_kernelILi16ELi2E7ik_bf16EE\|Function Name: .*fp8_gemv_rows_kernelILi8ELi2E7ik_bf16EE" k4fp8-out/compile-${PFX:-g}$g.log | grep -E "Function|VGPRs:|Scratch|Occupancy" | sed "s/.*remark: *//;s/ \[-R.*//;s/Function Name: _ZN12_GLOBAL__N_120//"
done
