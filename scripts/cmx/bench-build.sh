#!/usr/bin/env bash
# Lane cmx: compile the MXFP4 GEMV-vs-GEMM microbench on spark1 (run under fleetq --load cpu):
#   fleetq run --host spark1 --load cpu --mem 4 --expect 5 --label "cmx: bench build" -- scripts/cmx/bench-build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git rev-parse --show-toplevel)"; HOST=${HOST:-spark1}
ssh -n -o BatchMode=yes $HOST "mkdir -p ~/intarsiaml-cmx/bench"
rsync -a --delete "$REPO/kernels/" "$HOST:intarsiaml-cmx/bench/kernels/"
rsync -a "$REPO/scripts/cmx/mxfp4bench/" "$HOST:intarsiaml-cmx/bench/src/"
ssh -n -o BatchMode=yes $HOST 'export PATH=/usr/local/cuda/bin:$PATH; cd ~/intarsiaml-cmx/bench && nvcc -O3 -arch=sm_121 -std=c++17 -I kernels -I kernels/cuda -o bench src/bench.cu kernels/cuda/mxfp4.cu kernels/cuda/mxfp4_gemm.cu 2>&1 | grep -v "Warning: Cannot tell what pointer" | tail -30; ls -la bench'
