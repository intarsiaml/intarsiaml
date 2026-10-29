#!/usr/bin/env bash
# Lane conv16: the engine-level batched == solo gate on the 35B (real_35b_batched*, 4290 pairs), dev test binary, on spark1:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 35 --expect 10 --label "conv16: batch_cuda_real 35B" -- scripts/conv16/batch-real.sh [bin]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
B=${1:-dev}
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-conv16/src/crates/intarsia-models && INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 ~/intarsiaml-conv16/bin/batch_cuda_real-$B real_35b_batched --test-threads=1 --nocapture 2>&1 | grep -aE '^test |test result|panicked|FAIL|identical|pairs|row-table path' | tail -30"
