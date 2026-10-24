#!/usr/bin/env bash
# Lane c8k, spark side: the real `intarsia qualify` stages acc-short, acc-long and acc-4k (the CI-equivalent accuracy gates, reuse gates
# at their own chunk choice, committed continue oracles) for the 35B CUDA recipe, on a tree whose recipe chunk is 2048 (c8k-main) or
# 8192 (c8k-r8192, the Spark copy with the recipe override). Inside a fleetq slot (--quiet gpu,cpu).
# Usage: qualify.sh <outname> <tree>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; O=$P/out/$1; T=$2; mkdir -p $O
export PATH=/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
B=$P/$T/target/acc/release/intarsia
unset INTARSIA_CUDA_FP8_GEMM_LT
SKIP=""; for s in doctor recipe plan kernels gate acc-format acc-oracle bench envelope bench-long ab tp; do SKIP="$SKIP --skip $s"; done
echo "== recipe chunk of the tree: $(grep -n '^prefill_chunk' $P/$T/recipes/qwen3_5_moe.toml | head -2 | tr '\n' ' ')"
timeout 3400 $B qualify --model $M35 --backend cuda --device 0 --mem-budget 45 --golden-dir $P/pf-golden --model-name qwen36-35b-a3b-nvfp4 --out $O/report --no-unim $SKIP \
  --stage-timeout-min 40 < /dev/null > $O/qualify.log 2>&1
echo "qualify exit $?"
grep -aE "^ *(acc-|PASS|FAIL|SKIP|ERROR|MISSING)|acc-(short|long|4k)|prefill chunk|verdict" $O/qualify.log | cut -c1-240 | head -40
