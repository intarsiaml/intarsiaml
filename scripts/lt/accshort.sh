#!/usr/bin/env bash
# Lane lt, spark side: qualify's acc-short stage by hand (`accuracy --mode both` on the 35B short vLLM golden against the
# recorded CUDA baseline) per arm, to put lt's baseline-gate move next to other accumulation orders. Inside a fleetq slot.
# Usage: accshort.sh <outname> <arm> ...   arm = name:tree[:ENV=V,ENV=V]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; OUT=$P/out/$1; shift
export PATH=/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$P/pf-golden
mkdir -p $OUT
for spec in "$@"; do
  IFS=: read -r name tree envs <<< "$spec"; envs=${envs//,/ }
  B=$P/$tree/target/acc/release/intarsia
  echo "== acc-short $name env='$envs' $(date +%T)"
  env $envs timeout 1500 $B accuracy --model $M35 --backend cuda --golden $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --max-seq 16384 --mem-budget 40 \
    --baseline $GD/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.baseline.json --allow-dirty --allow-unstamped < /dev/null > $OUT/accshort-$name.log 2>&1
  echo "exit $?"; grep -aE "mean KL|tail-bucket KL|per-prompt gate:|verdict" $OUT/accshort-$name.log | cut -c1-200
done
echo "== done $(date +%T)"
