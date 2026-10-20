#!/usr/bin/env bash
# Round-6 window 2, spark1 side (vLLM down, INSIDE scripts/cuda-decode/window.sh, driven by scripts/cuda-prefill4/wC-localeval.sh
# with ARMS=dev): 1. the opt-in arm c4096 (dev with `serve --prefill-chunk 4096`, not bitwise) against dev, `bench speed
# --at-most`, ROUNDS (2) rounds (output ~/intarsia-decfuse/out/r6B2/ab); 2. one `serve` of dev for the driver's localeval run.
set -u
D=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export BENCH_ARGS="--at-most"
if [[ "${SKIP_AB:-0}" != 1 ]]; then
  echo "== speed A/B dev vs c4096 $(date +%T)"
  TREE=r6-dev OUTNAME=r6B2 ROUNDS=${ROUNDS:-2} ARMS="dev:r6-dev:INTARSIA_X=0 c4096:r6-dev:SERVE_ARGS=--prefill-chunk=4096" \
    W_BUDGET_SEC=${W_BUDGET_SEC:-500} bash $D/cuda-prefill4/wB-ab.sh
fi
echo "== localeval server dev $(date +%T)"
TREE=r6-dev OUTNAME=r6C bash $D/cuda-prefill4/wC-serve.sh "dev:INTARSIA_X=0"
