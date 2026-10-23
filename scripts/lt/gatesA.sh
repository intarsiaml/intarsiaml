#!/usr/bin/env bash
# Lane lt, spark side, flip PR gates part A: re-record the acc-short CUDA baseline from the lt build (default env, the
# original recording's flags), acc-short against it, and the three KL goldens (prefill, chunk 2048). Inside a fleetq slot.
# Usage: gatesA.sh <outname> <tree>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; O=$P/out/$1; T=$2; mkdir -p $O
export PATH=/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$P/pf-golden; B=$P/$T/target/acc/release/intarsia
unset INTARSIA_CUDA_FP8_GEMM_LT
echo "== record acc-short baseline $(date +%T)"
timeout 1500 $B accuracy --model $M35 --backend cuda --device 0 --mem-budget 40 --golden $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both \
  --decode-positions 16 --max-seq 4096 --write-baseline $O/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.baseline.json < /dev/null > $O/record.log 2>&1
echo "exit $?"; grep -aE "this build|wrote baseline|verdict" $O/record.log | cut -c1-200
echo "== acc-short against the new baseline $(date +%T)"
timeout 1500 $B accuracy --model $M35 --backend cuda --device 0 --mem-budget 40 --golden $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both \
  --decode-positions 16 --max-seq 4096 --baseline $O/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.baseline.json ${PRE_CHANGE:+--pre-change $PRE_CHANGE} < /dev/null > $O/accshort.log 2>&1
echo "exit $?"; grep -aE "^  mean KL|per-prompt gate:|verdict" $O/accshort.log | cut -c1-200
for g in qwen36-35b-a3b-nvfp4-vllm-top20 qwen36-35b-a3b-nvfp4-vllm-long-top20 qwen36-35b-a3b-nvfp4-vllm-4k-top20; do
  echo "== KL golden $g chunk 2048 $(date +%T)"
  timeout 1500 $B accuracy --model $M35 --backend cuda --golden $GD/$g.safetensors --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk 2048 \
    --allow-dirty < /dev/null > $O/kl-$g.log 2>&1
  echo "exit $?"; grep -aE "^all |KL\(21-way\)|floor verdict|golden gate" $O/kl-$g.log | tail -4 | cut -c1-200
done
echo "== done $(date +%T)"
