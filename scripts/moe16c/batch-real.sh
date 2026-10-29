#!/usr/bin/env bash
# Lane msd: the engine-level batched == solo gate on the 35B (G-C of docs/results/2026-10-23-rb2-cuda-batching.md), on spark1.
#   fleetq run --host spark1 --load cpu --mem 16 --expect 25 --label "msd: build batch_cuda_real" -- scripts/moe16c/batch-real.sh build
#   fleetq run --host spark1 --mem 35 --expect 25 --label "msd: batch_cuda_real 35B" -- scripts/moe16c/batch-real.sh run
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}; MODE=$1
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
if [ "$MODE" = build ]; then
  ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-moe16c/src && CARGO_BUILD_JOBS=16 cargo test -p intarsia-models --release --test batch_cuda_real --no-run 2>&1 | tail -4"
else
  ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-moe16c/src && INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real real_35b_batched -- --test-threads=1 --nocapture 2>&1 | grep -aE '^test |test result|panicked|FAIL|identical|pairs' | tail -30"
fi
