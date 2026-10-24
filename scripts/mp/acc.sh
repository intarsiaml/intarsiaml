#!/usr/bin/env bash
# Lane mp, spark side: `intarsia accuracy` of the 35B on CUDA per arm: prefill KL on the short, long and 4k goldens and the
# prefix-reuse gate. Inside a fleetq slot. Usage: acc.sh <outname> <arm> ...   arm = name:tree:chunk[:ENV=V,ENV=V]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; OUT=$P/out/$1; shift
export PATH=/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$P/pf-golden
mkdir -p $OUT
for spec in "$@"; do
  IFS=: read -r name tree chunk envs <<< "$spec"; envs=${envs//,/ }
  B=$P/$tree/target/acc/release/intarsia
  for g in qwen36-35b-a3b-nvfp4-vllm-top20 qwen36-35b-a3b-nvfp4-vllm-long-top20 qwen36-35b-a3b-nvfp4-vllm-4k-top20; do
    t=${g#qwen36-35b-a3b-nvfp4-vllm-}; t=${t%-top20}; [[ $g == *vllm-top20 ]] && t=short
    echo "== acc $name $t chunk=$chunk env='$envs' $(date +%T)"
    env $envs timeout 1500 $B accuracy --model $M35 --backend cuda --golden $GD/$g.safetensors --mode prefill --max-seq 16384 --mem-budget 40 \
      --prefill-chunk $chunk --allow-dirty < /dev/null > $OUT/acc-$name-$t.log 2>&1
    echo "exit $?"; grep -aE "KL|top-1|verdict|PASS|FAIL" $OUT/acc-$name-$t.log | head -8 | cut -c1-200
  done
  for rc in 1024 $chunk; do
    echo "== reuse $name chunk=$rc env='$envs' $(date +%T)"
    env $envs timeout 1500 $B accuracy --model $M35 --backend cuda --golden $GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode reuse --max-seq 16384 --mem-budget 40 \
      --prefill-chunk $rc --allow-dirty < /dev/null > $OUT/reuse-$name-c$rc.log 2>&1
    echo "exit $?"; grep -aE "chat_reuse|prefix-reuse|PASS|FAIL" $OUT/reuse-$name-c$rc.log | head -8 | cut -c1-200
  done
done
echo "== done $(date +%T)"
