#!/usr/bin/env bash
# Lane S1 spark2 window body (vLLM down, INSIDE scripts/s1/window.sh with LOCALEVAL_ARMS="base dev cub"):
#   1. the in-engine A/B (scripts/s1/wAB.sh: `intarsia bench speed --at-most`, 1k / 4k / 16k + decode, 3 interleaved rounds,
#      arms base / dev / cub);
#   2. one `serve` per arm for localeval speed (scripts/cuda-prefill4/wC-serve.sh, READY protocol), the harness the Q1 vLLM
#      numbers came from.
# STEPS (default "ab le") picks the parts.
set -u
P=$HOME/intarsia-decfuse
S=$P/s1-dev/scripts
STEPS=${STEPS:-ab le}
export OUTNAME=${OUTNAME:-s1W}
if [[ " $STEPS " == *" ab "* ]]; then
  AB_ARMS=${AB_ARMS:-"base:s1-base dev:s1-dev cub:s1-dev:INTARSIA_CUDA_FP8_CUBLAS=1"} W_BUDGET_SEC=${W_BUDGET_SEC:-2400} bash $S/s1/wAB.sh
fi
if [[ " $STEPS " == *" le "* ]]; then
  TREE=s1-base ARM_MAX_SEC=600 bash $S/cuda-prefill4/wC-serve.sh base
  TREE=s1-dev ARM_MAX_SEC=600 bash $S/cuda-prefill4/wC-serve.sh dev cub:INTARSIA_CUDA_FP8_CUBLAS=1
fi
echo "== wALL done $(date +%T)"
