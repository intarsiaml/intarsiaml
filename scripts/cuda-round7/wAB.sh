#!/usr/bin/env bash
# Round-7 speed windows, spark1 side (vLLM down, INSIDE scripts/cuda-round7/window.sh):
#   wAB.sh ab   accuracy extras (scripts/cuda-round7/acc2.sh), then the interleaved A/B: `intarsia bench speed --at-most`
#               against a fresh `serve` per arm (scripts/cuda-prefill4/wB-ab.sh -> scripts/cuda-decode/w-ab.sh), sizes
#               1024 / 4096 / 16384, gen 128, 2 reps per size, ROUNDS (3) rounds, order reversed in even rounds. Arms:
#               base = tree r7-base (origin/main as is); lt = r7-dev with INTARSIA_CUDA_FP8_GEMM_LT=1; lt4k = lt with
#               `serve --prefill-chunk 4096`. Output ~/intarsia-decfuse/out/r7B/ab
#   wAB.sh le   one `serve` per localeval arm (lt, lt4k) for the Mac driver (window.sh LOCALEVAL_ARMS), done markers in out/r7C.
set -u
D=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export BENCH_ARGS="--at-most"
case "${1:?ab|le}" in
  ab)
    echo "== accuracy extras $(date +%T)"
    bash $D/cuda-round7/acc2.sh
    echo "== speed A/B $(date +%T)"
    TREE=r7-dev OUTNAME=r7B ROUNDS=${ROUNDS:-3} W_BUDGET_SEC=${W_BUDGET_SEC:-900} \
      ARMS="${AB_ARMS:-base:r7-base:INTARSIA_X=0 lt:r7-dev:INTARSIA_CUDA_FP8_GEMM_LT=1 lt4k:r7-dev:INTARSIA_CUDA_FP8_GEMM_LT=1,SERVE_ARGS=--prefill-chunk=4096}" \
      bash $D/cuda-prefill4/wB-ab.sh
    ;;
  le)
    echo "== localeval servers $(date +%T)"
    TREE=r7-dev OUTNAME=r7C ARM_MAX_SEC=420 bash $D/cuda-prefill4/wC-serve.sh "lt:INTARSIA_CUDA_FP8_GEMM_LT=1" \
      "lt4k:INTARSIA_CUDA_FP8_GEMM_LT=1,SERVE_ARGS=--prefill-chunk=4096"
    ;;
esac
