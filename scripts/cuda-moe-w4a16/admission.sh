#!/usr/bin/env bash
# Admission sweep of the CUDA backend for the MoE experts kernels against the cpu-ref oracle, shapes from the 35B config
# (vLLM UP, small slot):  fleetq run --host spark1 --mem 12 --label "w4a16 admission" -- scripts/cuda-moe-w4a16/admission.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsia-decfuse/w4-dev && ./target/acc/release/intarsia kernels test --backend cuda --model $M --op moe_experts -v 2>&1 | tail -80"
