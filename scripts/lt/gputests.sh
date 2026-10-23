#!/usr/bin/env bash
# Lane lt, spark side: build (target/test) and run the CUDA lt GEMM tests of a tree, optionally `kernels test --quick`.
# Inside a fleetq slot. Usage: gputests.sh <tree> [filter] [quick]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
cd ~/intarsia-decfuse/$1 || exit 1
F=${2:-fp8_gemm_lt}
[ -d target/test ] || cp -a target/acc target/test
echo "== cargo test --test quantmoe $F $(date +%T)"
CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} CARGO_TARGET_DIR=target/test INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test quantmoe $F -- --test-threads=1 --nocapture < /dev/null 2>&1 | grep -aE "^test |test result|panicked|error|rel_l2|differ|lt:" | cut -c1-240
if [ "${3:-}" = quick ]; then
  echo "== cargo test -p intarsia-cli --test docs_env_vars $(date +%T)"
  CARGO_TARGET_DIR=target/test cargo test --release -p intarsia-cli --test docs_env_vars < /dev/null 2>&1 | grep -aE "^test |test result|panicked|error" | cut -c1-200
  for e in "" "INTARSIA_CUDA_FP8_GEMM_LT=1"; do
    echo "== kernels test --quick env='$e' $(date +%T)"
    env $e timeout 1500 target/acc/release/intarsia kernels test --backend cuda --quick < /dev/null 2>&1 | tail -6 | cut -c1-200
  done
fi
echo "== done $(date +%T)"
