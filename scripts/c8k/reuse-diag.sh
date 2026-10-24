#!/usr/bin/env bash
# Lane c8k, spark side: diagnose the chunk-2048 s6000 continue FAIL of long_docs_12k (reuse-L2048 in ru2): the reuse gate at chunks 2048 and
# 1024 on the long golden for that prompt only, with --reuse-diag (warm vs cold row comparison, first divergence). Inside a fleetq slot.
# Usage: reuse-diag.sh <outname> <tree>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; O=$P/out/$1; T=$2; mkdir -p $O
export PATH=/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
B=$P/$T/target/acc/release/intarsia; GD=$P/pf-golden-c8k
unset INTARSIA_CUDA_FP8_GEMM_LT
for C in 1024 2048; do
  for pr in long_docs_12k long_code_12k; do
    echo "== reuse $pr chunk $C $(date +%T)"
    timeout 1500 $B accuracy --model $M35 --backend cuda --golden $GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode reuse --max-seq 16384 --mem-budget 40 \
      --prefill-chunk $C --allow-dirty --reuse-prompt $pr --reuse-diag < /dev/null > $O/diag-$pr-c$C.log 2>&1
    echo "exit $?"; grep -aE "^$pr +(continue|restore|aligned)|dump:|diag|warm_pf|first div|oracle-relative" $O/diag-$pr-c$C.log | cut -c1-260 | head -14
  done
done
echo "== done $(date +%T)"
