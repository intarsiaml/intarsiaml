#!/usr/bin/env bash
# Window B+C (spark1, vLLM down, INSIDE scripts/cuda-decode/window.sh, driven by scripts/cuda-prefill3/wC-localeval.sh):
#   1. wB-ab.sh: interleaved speed A/B of AB_ARMS, 3 rounds (budget W_BUDGET_SEC)
#   2. wC-serve.sh: one `serve` per LE_ARMS entry for the localeval runs of the driver
# One binary (tree p3-dev) for every arm; only the environment differs. Edit the arm lists here, not on the command line.
set -u
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AB_ARMS=${AB_ARMS:-"base:p3-dev:INTARSIA_CUDA_FP8_GEMM_REG=0 lanes:p3-dev:INTARSIA_CUDA_FP8_GEMM_REG=0,INTARSIA_CUDA_GDN_LANES=1 reg:p3-dev:INTARSIA_X=0"}
LE_ARMS=${LE_ARMS:-"dev:INTARSIA_X=0"}
echo "== W-B speed A/B $(date +%T)"
ARMS="$AB_ARMS" W_BUDGET_SEC=${W_BUDGET_SEC:-680} bash $D/wB-ab.sh
echo "== W-C localeval servers $(date +%T)"
bash $D/wC-serve.sh $LE_ARMS
