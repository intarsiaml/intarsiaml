#!/usr/bin/env bash
# Lane mp, spark side, model-accuracy gates with the opt-in INTARSIA_CUDA_MOE_W4A16_PF=1: acc-short against main's committed
# CUDA baseline (golden/baselines, recorded from c37a8d3 = the lt build) and the three KL goldens at prefill chunk 2048.
# Inside a fleetq slot. Usage: gatesA.sh <outname> <tree> <baseline.json> [pf]    pf=0 runs the same gates without the knob (control)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; O=$P/out/$1; T=$2; BL=$3; PF=${4:-1}; mkdir -p $O
export PATH=/usr/local/cuda/bin:$PATH INTARSIA_CUDA_MOE_W4A16_PF=$PF
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$P/pf-golden; B=$P/$T/target/acc/release/intarsia
echo "== acc-short vs main baseline, PF=$PF $(date +%T)"
timeout 1500 $B accuracy --model $M35 --backend cuda --device 0 --mem-budget 40 --golden $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both \
  --decode-positions 16 --max-seq 4096 --baseline $BL --pre-change c37a8d3 < /dev/null > $O/accshort.log 2>&1
echo "exit $?"; grep -aE "^  mean KL|per-prompt gate:|verdict|Error" $O/accshort.log | cut -c1-220
for g in qwen36-35b-a3b-nvfp4-vllm-top20 qwen36-35b-a3b-nvfp4-vllm-long-top20 qwen36-35b-a3b-nvfp4-vllm-4k-top20; do
  echo "== KL golden $g chunk 2048 PF=$PF $(date +%T)"
  timeout 1500 $B accuracy --model $M35 --backend cuda --golden $GD/$g.safetensors --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk 2048 \
    --allow-dirty < /dev/null > $O/kl-$g.log 2>&1
  echo "exit $?"; grep -aE "^all |KL\(21-way\)|floor verdict|golden gate" $O/kl-$g.log | tail -4 | cut -c1-200
done
echo "== done $(date +%T)"
