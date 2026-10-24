#!/usr/bin/env bash
# Lane c8k, spark side: batch_cuda_real (default tests) and the ignored 35B long-prompt engine test on a tree whose 35B CUDA recipe
# chunk was set to 8192 in the Spark copy only (scripts/c8k/build.sh RECIPE_CHUNK=8192 TESTS=1). The busy chunk follows the solo chunk,
# so batched prefill runs at 8192 too. Inside a fleetq slot (--quiet gpu,cpu). Usage: gatesB.sh <outname> <tree>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; O=$P/out/$1; T=$2; mkdir -p $O
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
cd $P/$T || exit 1
echo "== recipe in the tree: $(grep -n '^prefill_chunk' recipes/qwen3_5_moe.toml | tr '\n' ' ')"
echo "== batch_cuda_real $(date +%T)"
INTARSIA_QWEN36_DIR=$M35 INTARSIA_QWEN3_DIR=$M8 INTARSIA_REQUIRE_GPU=1 CARGO_TARGET_DIR=target/test timeout 3000 cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 < /dev/null > $O/batch.log 2>&1
echo "exit $?"; grep -aE "^test |test result|panicked|prefill chunk" $O/batch.log | cut -c1-200
echo "== batch_cuda_real --ignored long prompts $(date +%T)"
INTARSIA_QWEN36_DIR=$M35 INTARSIA_QWEN3_DIR=$M8 INTARSIA_REQUIRE_GPU=1 CARGO_TARGET_DIR=target/test timeout 3000 cargo test --release -p intarsia-models --test batch_cuda_real -- --ignored --test-threads=1 --nocapture real_35b_batched_engine_long_prompts_equal_the_local_engine < /dev/null > $O/batch-long.log 2>&1
echo "exit $?"; grep -aE "^test |test result|panicked|LONGENG|prefill chunk|busy" $O/batch-long.log | cut -c1-220
echo "== done $(date +%T)"
