#!/usr/bin/env bash
# Round-5 vLLM-down window, spark1 side (INSIDE scripts/cuda-decode/window.sh, driven by scripts/cuda-prefill4/wC-localeval.sh):
#   1. scripts/cuda-prefill4/wB-ab.sh: interleaved speed A/B of AB_ARMS, ROUNDS (3) rounds (budget W_BUDGET_SEC); SKIP_AB=1 skips
#   2. scripts/cuda-prefill4/wC-serve.sh: one `serve` per LE_ARMS entry for the driver's localeval runs
# base = tree r5-base (origin/main as is); dev = tree r5-f8 with the default bf16 KV cache (bf16 attention SASS identical
# to base); f8 = tree r5-f8 with the opt-in fp8 KV cache (`serve --kv-dtype fp8_e4m3`, SERVE_ARGS in the arm spec).
set -u
D=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
AB_ARMS=${AB_ARMS:-"base:r5-base:INTARSIA_X=0 dev:r5-f8:INTARSIA_X=0 f8:r5-f8:SERVE_ARGS=--kv-dtype=fp8_e4m3"}
LE_ARMS=${LE_ARMS:-"f8:SERVE_ARGS=--kv-dtype=fp8_e4m3"}
if [[ "${SKIP_AB:-0}" != 1 ]]; then
  echo "== speed A/B $(date +%T)"
  TREE=r5-f8 OUTNAME=${OUTNAME:-r5B} ARMS="$AB_ARMS" W_BUDGET_SEC=${W_BUDGET_SEC:-660} bash $D/cuda-prefill4/wB-ab.sh
fi
echo "== localeval servers $(date +%T)"
TREE=${LE_TREE:-r5-f8} OUTNAME=${LE_OUT:-r5C} bash $D/cuda-prefill4/wC-serve.sh $LE_ARMS
