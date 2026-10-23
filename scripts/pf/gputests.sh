#!/usr/bin/env bash
# Lane pf, spark side: the CUDA GPU tests that cover the w4a16 MoE change, in the tree's own test target dir. Inside a fleetq slot.
# Usage: gputests.sh <tree>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
cd ~/intarsia-decfuse/$1 || exit 1
for t in moe_w4a16 moe_w4a16_ip quantmoe; do
  echo "== cargo test --test $t $(date +%T)"
  CARGO_TARGET_DIR=target/test INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test $t -- --test-threads=1 2>&1 | grep -aE "^test |test result|panicked|error" | cut -c1-200
done
echo "== cargo test -p intarsia-cli --test docs_env_vars $(date +%T)"
CARGO_TARGET_DIR=target/test cargo test --release -p intarsia-cli --test docs_env_vars 2>&1 | grep -aE "^test |test result|panicked|error" | cut -c1-200
echo "== kernels test (admission, default env) $(date +%T)"
timeout 1500 target/acc/release/intarsia kernels test --backend cuda --quick < /dev/null 2>&1 | tail -6 | cut -c1-200
echo "== done $(date +%T)"
