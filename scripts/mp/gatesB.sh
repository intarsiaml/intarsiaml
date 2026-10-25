#!/usr/bin/env bash
# Lane mp, spark side: prefix-reuse gate at chunk 2048 (committed CPU continue oracles) and the CUDA batching gates
# (batch_cuda_real, plus the ignored long-prompt engine test), with INTARSIA_CUDA_MOE_W4A16_PF=1. Failures are counted, not grepped
# for success words. Inside a fleetq slot. Usage: gatesB.sh <outname> <tree> [pf]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; O=$P/out/$1; T=$2; PF=${3:-1}; mkdir -p $O
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH INTARSIA_CUDA_MOE_W4A16_PF=$PF
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
B=$P/$T/target/acc/release/intarsia
echo "== reuse gate chunk 2048 PF=$PF $(date +%T)"
timeout 1800 $B accuracy --model $M35 --backend cuda --golden $P/pf-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode reuse --max-seq 16384 --mem-budget 40 \
  --prefill-chunk 2048 --allow-dirty < /dev/null > $O/reuse.log 2>&1
echo "exit $?"; grep -aE "^\S+ +continue|prefix-reuse gate" $O/reuse.log | cut -c1-220
cd $P/$T
echo "== batch_cuda_real PF=$PF $(date +%T)"
INTARSIA_QWEN36_DIR=$M35 INTARSIA_QWEN3_DIR=$M8 INTARSIA_REQUIRE_GPU=1 CARGO_TARGET_DIR=target/test timeout 2400 cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 < /dev/null > $O/batch.log 2>&1
echo "exit $?"; grep -aE "^test |test result|panicked" $O/batch.log | cut -c1-200
echo "FAILED count: $(grep -ac '^test .* FAILED' $O/batch.log)"
echo "== batch_cuda_real --ignored long prompts PF=$PF $(date +%T)"
INTARSIA_QWEN36_DIR=$M35 INTARSIA_QWEN3_DIR=$M8 INTARSIA_REQUIRE_GPU=1 CARGO_TARGET_DIR=target/test timeout 2400 cargo test --release -p intarsia-models --test batch_cuda_real -- --ignored --test-threads=1 real_35b_batched_engine_long_prompts_equal_the_local_engine < /dev/null > $O/batch-long.log 2>&1
echo "exit $?"; grep -aE "^test |test result|panicked" $O/batch-long.log | cut -c1-200
echo "FAILED count: $(grep -ac '^test .* FAILED' $O/batch-long.log)"
echo "== done $(date +%T)"
