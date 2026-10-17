#!/usr/bin/env bash
# Which NVFP4 MoE / GEMM backend the running vLLM (container missioncntrl-guidance) picked: read-only log grep, no GPU use:
#   fleetq run --host spark1 --mem 1 --label "nvfp4-mma vllm backend log" -- scripts/cuda-moe-nvfp4/vllm-backend.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
ssh -n -o BatchMode=yes "${HOST:-spark1}" "FLEETQ_JOB=$FLEETQ_JOB docker logs missioncntrl-guidance 2>&1 | grep -E 'Initializing a V1 LLM engine|NvFp4 MoE backend|for NVFP4 GEMM|native support for FP4' | cut -c1-330"
