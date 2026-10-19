#!/usr/bin/env bash
# Run the CUDA backend tests of a tree on spark1 with vLLM UP (correctness tests only: small buffers, no timing claims):
#   fleetq run --host spark1 --mem 4 --label "..." -- scripts/cuda-decode/gpu-test.sh <tree> <cargo test args after `cargo test --release -p intarsia-backend-cuda`>
#   e.g.  gpu-test.sh dev --test fused_decode --test admission -- --nocapture --test-threads=1
# FLEETQ_JOB (set by `fleetq run`) goes on the ssh.
# Then (unless RACECHECK=0) the shared-memory race check of every kernel in scripts/cuda-checks/race-targets.txt
# (scripts/cuda-checks/racecheck.sh): a dropped __syncthreads / __syncwarp that leaves the bits unchanged fails here.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
NAME=${1:?tree name}; shift
HOST=${HOST:-spark1}
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsia-decfuse/$NAME && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda $*"
if [[ "${RACECHECK:-1}" != 0 ]]; then
  echo "== racecheck (scripts/cuda-checks/race-targets.txt)"
  HOST=$HOST "$S/../cuda-checks/racecheck.sh" "$NAME"
fi
