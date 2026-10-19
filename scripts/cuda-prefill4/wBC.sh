#!/usr/bin/env bash
# Window B+C (spark1, vLLM down, INSIDE scripts/cuda-decode/window.sh, driven by scripts/cuda-prefill4/wC-localeval.sh):
#   1. wB-ab.sh: interleaved speed A/B of AB_ARMS, 3 rounds (budget W_BUDGET_SEC)
#   2. wC-serve.sh: one `serve` per LE_ARMS entry for the localeval runs of the driver
# base = tree p4-base (origin/main 8638db03 as is); the dev arms are tree p4-dev, only the environment differs. Edit the arm lists here, not on the command line.
set -u
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AB_ARMS=${AB_ARMS:-"base:p4-base:INTARSIA_X=0 split:p4-dev:INTARSIA_X=0 f8:p4-dev:INTARSIA_CUDA_GDN_SPLIT=0,INTARSIA_CUDA_FP8_GEMM_F8=1"}
LE_ARMS=${LE_ARMS:-"dev:INTARSIA_X=0"}
echo "== W-B speed A/B $(date +%T)"
ARMS="$AB_ARMS" W_BUDGET_SEC=${W_BUDGET_SEC:-680} bash $D/wB-ab.sh
echo "== W-C localeval servers $(date +%T)"
bash $D/wC-serve.sh $LE_ARMS
