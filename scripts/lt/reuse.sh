#!/usr/bin/env bash
# Lane lt, spark side: the prefix-reuse gate (`accuracy --mode reuse`, 35B long golden, 5 prompts x continue/restore/aligned) per
# arm, without diagnostics. Inside a fleetq slot. Usage: reuse.sh <outname> <arm> ...   arm = name:tree:chunk[:ENV=V,ENV=V]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; OUT=$P/out/$1; shift
export PATH=/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
mkdir -p $OUT
for spec in "$@"; do
  IFS=: read -r name tree chunk envs <<< "$spec"; envs=${envs//,/ }
  B=$P/$tree/target/acc/release/intarsia
  echo "== reuse $name chunk=$chunk env='$envs' $(date +%T)"
  env $envs timeout 1800 $B accuracy --model $M35 --backend cuda --golden $P/pf-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode reuse --max-seq 16384 --mem-budget 40 \
    --prefill-chunk $chunk --allow-dirty ${EXTRA_ARGS:-} < /dev/null > $OUT/reuse-$name.log 2>&1
  echo "exit $?"; grep -aE "^\S+ +continue|prefix-reuse gate" $OUT/reuse-$name.log | cut -c1-230
done
echo "== done $(date +%T)"
