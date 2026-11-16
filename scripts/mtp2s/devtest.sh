#!/usr/bin/env bash
# mtp2s: run a CUDA device test of the tree synced by build.sh on spark2 (default: the bitwise test of the fused GDN verify launches).
#   fleetq run --host spark2 --quiet gpu,cpu --mem 8 --expect 15 --label "mtp2s: device test" -- scripts/mtp2s/devtest.sh [test-name [extra cargo test args]]
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
T=${1:-gdn_verify_snap}; shift || true
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
ssh -n -o BatchMode=yes spark2 "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsia-mtp2s/src && env CARGO_BUILD_JOBS=4 MAX_JOBS=4 cargo test --release --locked -p intarsia-backend-cuda --test $T $* -- --nocapture 2>&1 | grep -v '^warning\|^ *|\|^ *-->\|^ *=\|^$' | tail -40"
