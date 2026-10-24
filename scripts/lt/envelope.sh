#!/usr/bin/env bash
# Lane lt, spark side: re-record the acc-short noise envelope from the build that recorded the baseline (same flags as the
# original envelope: --noise-perturb chunk), then acc-short against baseline + envelope as qualify runs it. Inside a fleetq slot.
# Usage: envelope.sh <outname> <tree> <baseline.json>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; O=$P/out/$1; mkdir -p $O; export PATH=/usr/local/cuda/bin:$PATH; unset INTARSIA_CUDA_FP8_GEMM_LT
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
B=$P/$2/target/acc/release/intarsia; G=$P/pf-golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
timeout 1500 $B accuracy --model $M35 --backend cuda --device 0 --mem-budget 40 --golden $G --mode both --decode-positions 16 --max-seq 4096 \
  --noise-ref $3 --noise-perturb chunk --write-noise-envelope $O/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.envelope.json --allow-same-code < /dev/null > $O/envelope.log 2>&1
echo "envelope exit $?"; grep -aE "envelope|Error" $O/envelope.log | tail -3 | cut -c1-220
timeout 1500 $B accuracy --model $M35 --backend cuda --device 0 --mem-budget 40 --golden $G --mode both --decode-positions 16 --max-seq 4096 \
  --baseline $3 --noise-envelope $O/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.envelope.json --pre-change ${4:-c37a8d3} --allow-same-code < /dev/null > $O/accshort-env.log 2>&1
echo "acc-short (baseline + envelope) exit $?"; grep -aE "^  mean KL|per-prompt gate:|verdict|Error" $O/accshort-env.log | cut -c1-220
