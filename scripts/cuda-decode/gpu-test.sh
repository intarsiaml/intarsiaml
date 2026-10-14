#!/usr/bin/env bash
# Run the CUDA backend tests of a tree on spark1 with vLLM UP (correctness tests only: small buffers, no timing claims):
#   fleetq run --host spark1 --mem 4 --label "..." -- scripts/cuda-decode/gpu-test.sh <tree> <cargo test args after `cargo test --release -p intarsia-backend-cuda`>
#   e.g.  gpu-test.sh dev --test fused_decode --test admission -- --nocapture --test-threads=1
# FLEETQ_JOB (set by `fleetq run`) goes on the ssh.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
NAME=${1:?tree name}; shift
HOST=${HOST:-spark1}
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsia-decfuse/$NAME && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda $*"
