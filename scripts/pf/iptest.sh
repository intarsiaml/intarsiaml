#!/usr/bin/env bash
# Lane pf, spark side: the in-place-accumulation MoE test with its numbers visible. Inside a fleetq slot. Usage: iptest.sh <tree>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
cd ~/intarsia-decfuse/$1 || exit 1
CARGO_TARGET_DIR=target/test INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test moe_w4a16_ip -- --test-threads=1 --nocapture 2>&1 | grep -aE "^T=|^test |test result|panicked|assert|in-place|left|right" | cut -c1-260
