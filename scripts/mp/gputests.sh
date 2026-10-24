#!/usr/bin/env bash
# Lane mp, spark side: the CUDA GPU tests that cover the w4a16 MoE change, in the tree's own test target dir. Inside a fleetq slot.
# Usage: gputests.sh <tree>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
cd ~/intarsia-decfuse/$1 || exit 1
[ -d target/test ] || cp -a target/acc target/test
for t in moe_w4a16 quantmoe moe_nvfp4; do
  echo "== cargo test --test $t $(date +%T)"
  CARGO_TARGET_DIR=target/test INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test $t -- --test-threads=1 2>&1 | grep -aE "^test |test result|panicked|error" | cut -c1-200
done
echo "== cargo test -p intarsia-cli --test docs_env_vars $(date +%T)"
CARGO_TARGET_DIR=target/test cargo test --release -p intarsia-cli --test docs_env_vars 2>&1 | grep -aE "^test |test result|panicked|error" | cut -c1-200
for e in X=1 INTARSIA_CUDA_MOE_W4A16_PF=1; do
  echo "== kernels test --quick env=$e $(date +%T)"
  env $e timeout 1500 target/acc/release/intarsia kernels test --backend cuda --quick < /dev/null > $HOME/intarsia-decfuse/out/mp-kt-$e.txt 2>&1
  echo "exit $? FAIL lines $(grep -ac FAIL $HOME/intarsia-decfuse/out/mp-kt-$e.txt)"; tail -4 $HOME/intarsia-decfuse/out/mp-kt-$e.txt | cut -c1-200
done
echo "== done $(date +%T)"
