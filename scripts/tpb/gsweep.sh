#!/usr/bin/env bash
# Lane TPB, ON tatooine INSIDE a GPU window (drive.sh action gsweep): a8 GEMM variants. GEMMV = ";"-separated list of INTARSIA_HIP_MXFP4_A8_TILE values
# ("" or "default" = the shipped table); for each: the correctness tests (CHECK=1) and the 27B tp2 timing (timing_a8_tp2), one process per value
# (the kernel reads the knob once). Output "== variant" then the TIME lines / test results.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
IFS=';' read -ra V <<< "${GEMMV:-default}"
for v in "${V[@]}"; do
  e="A=1"; [ -n "$v" ] && [ "$v" != default ] && e="INTARSIA_HIP_MXFP4_A8_TILE=$v"
  echo "== variant ${v:-default}"
  if [ "${CHECK:-0}" = 1 ] && [[ $v != *,*,*,*,*,[12] ]]; then
    G=0 gpu 600 env $e cargo test --release -p intarsia-backend-hip --test mxfp4_a8 -- a8_kernels_match a8_mma_rows a8_mma_wide 2>&1 | grep -aE "^test |test result|panicked|rel_l2|differ" | head -12
  fi
  G=0 gpu 600 env $e cargo test --release -p intarsia-backend-hip --test mxfp4_a8 -- --ignored --nocapture timing_a8_tp2 2>&1 | grep -aE "^TIME|panicked|error" | head -14
done
