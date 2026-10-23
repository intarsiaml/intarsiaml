#!/usr/bin/env bash
# Lane pf, spark side: build the intarsia-backend-cuda test binaries of a tree in its own target dir (target/test). Inside a fleetq slot.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
cd ~/intarsia-decfuse/$1 || exit 1
[ -d target/test ] || cp -a target/acc target/test
CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} CARGO_TARGET_DIR=target/test cargo test --release -p intarsia-backend-cuda --no-run 2>&1 | grep -E '^error|Finished|Executable' | tail -30
