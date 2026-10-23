#!/usr/bin/env bash
# Lane lt, spark side: acc-short (mode both, the recording's flags) against a baseline file, with --pre-change <sha>. Inside a fleetq slot.
# Usage: accshort2.sh <outname> <tree> <baseline.json> <pre-change sha> [same]  (same: --allow-same-code, a reproducibility check)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; O=$P/out/$1; mkdir -p $O; export PATH=/usr/local/cuda/bin:$PATH; unset INTARSIA_CUDA_FP8_GEMM_LT
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
timeout 1500 $P/$2/target/acc/release/intarsia accuracy --model $M35 --backend cuda --device 0 --mem-budget 40 --golden $P/pf-golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors \
  --mode both --decode-positions 16 --max-seq 4096 --baseline $3 --pre-change $4 ${5:+--allow-same-code} < /dev/null > $O/accshort.log 2>&1
echo "exit $?"; grep -aE "^  mean KL|per-prompt gate:|verdict|Error" $O/accshort.log | cut -c1-220
