#!/usr/bin/env bash
# Lane lt, spark side, flip PR gates part B: prefix-reuse gate at chunk 2048 (committed continue oracles), the CUDA lt unit /
# GPU tests, `kernels test --quick` (default = lt, LT=0), and the CUDA batching gates (batch_cuda_real, plus the ignored
# long-prompt engine test). Inside a fleetq slot. Usage: gatesB.sh <outname> <tree>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; O=$P/out/$1; T=$2; mkdir -p $O
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
B=$P/$T/target/acc/release/intarsia
unset INTARSIA_CUDA_FP8_GEMM_LT
echo "== reuse gate chunk 2048 $(date +%T)"
timeout 1800 $B accuracy --model $M35 --backend cuda --golden $P/pf-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode reuse --max-seq 16384 --mem-budget 40 \
  --prefill-chunk 2048 --allow-dirty < /dev/null > $O/reuse.log 2>&1
echo "exit $?"; grep -aE "^\S+ +continue|prefix-reuse gate" $O/reuse.log | cut -c1-220
cd $P/$T
echo "== cargo test lib + quantmoe lt tests $(date +%T)"
CARGO_TARGET_DIR=target/test INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --lib fp8_ < /dev/null 2>&1 | grep -aE "^test |test result|panicked" | cut -c1-200
CARGO_TARGET_DIR=target/test INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test quantmoe fp8_gemm_lt -- --test-threads=1 < /dev/null 2>&1 | grep -aE "^test |test result|panicked" | cut -c1-200
CARGO_TARGET_DIR=target/test INTARSIA_CUDA_FP8_GEMM_LT=0 cargo test --release -p intarsia-backend-cuda --lib fp8_ < /dev/null 2>&1 | grep -aE "test result|panicked" | sed 's/^/LT=0: /' | cut -c1-200
CARGO_TARGET_DIR=target/test cargo test --release -p intarsia-cli --test docs_env_vars < /dev/null 2>&1 | grep -aE "test result|panicked" | cut -c1-200
for e in X=1 INTARSIA_CUDA_FP8_GEMM_LT=0; do
  env $e timeout 900 $B kernels test --backend cuda --quick < /dev/null > $O/kt-$e.txt 2>&1
  echo "kernels test --quick env=$e: exit $? FAIL lines $(grep -c FAIL $O/kt-$e.txt), $(tail -3 $O/kt-$e.txt | grep -a 'kernels,')"
done
echo "== batch_cuda_real $(date +%T)"
INTARSIA_QWEN36_DIR=$M35 INTARSIA_QWEN3_DIR=$M8 INTARSIA_REQUIRE_GPU=1 CARGO_TARGET_DIR=target/test timeout 2400 cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 < /dev/null > $O/batch.log 2>&1
echo "exit $?"; grep -aE "^test |test result|panicked" $O/batch.log | cut -c1-200
echo "== batch_cuda_real --ignored long prompts $(date +%T)"
INTARSIA_QWEN36_DIR=$M35 INTARSIA_QWEN3_DIR=$M8 INTARSIA_REQUIRE_GPU=1 CARGO_TARGET_DIR=target/test timeout 2400 cargo test --release -p intarsia-models --test batch_cuda_real -- --ignored --test-threads=1 real_35b_batched_engine_long_prompts_equal_the_local_engine < /dev/null > $O/batch-long.log 2>&1
echo "exit $?"; grep -aE "^test |test result|panicked" $O/batch-long.log | cut -c1-200
echo "== done $(date +%T)"
