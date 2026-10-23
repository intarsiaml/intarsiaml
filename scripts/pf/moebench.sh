#!/usr/bin/env bash
# Lane pf, spark side: routed-expert microbench (tests/moe_nvfp4.rs bench_moe_nvfp4_real_shapes: 256 experts, top-8, H 2048, I 512,
# uniform random routing) per env combo. Inside a fleetq slot. Usage: moebench.sh <tree> <T list> <kernel list> <combo> ...
#   combo = name[:ENV=V,ENV=V]   kernel list = bf16,fp4,v2 subset
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
cd ~/intarsia-decfuse/$1 || exit 1; TS=$2; KS=$3; shift 3
for spec in "$@"; do
  IFS=: read -r name envs <<< "$spec"; envs=${envs//,/ }
  echo "== $name env='$envs' $(date +%T)"
  env $envs INTARSIA_BENCH_T=$TS INTARSIA_BENCH_KERNEL=$KS INTARSIA_REQUIRE_GPU=1 CARGO_TARGET_DIR=target/test \
    cargo test --release -p intarsia-backend-cuda --test moe_nvfp4 -- --ignored --nocapture --test-threads=1 bench_moe_nvfp4_real_shapes 2>&1 \
    | grep -aE "^[0-9]+ +moe_|panicked|error" | head -40
done
echo "== done $(date +%T)"
