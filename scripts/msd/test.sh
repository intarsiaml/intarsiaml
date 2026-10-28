#!/usr/bin/env bash
# Lane msd: build + run the CUDA backend tests of the synced tree on spark1 (after scripts/msd/build.sh synced it).
#   fleetq run --host spark1 --load cpu --expect 20 --label "msd: build tests" -- scripts/msd/test.sh build
#   fleetq run --host spark1 --mem 8 --expect 15 --label "msd: rows_exact" -- scripts/msd/test.sh run <test-args...>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}; MODE=$1; shift
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-msd/src && if [ $MODE = build ]; then CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test rows_exact --test rows_registry --test quick_coverage --no-run 2>&1 | tail -6; else INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-backend-cuda --release $* 2>&1 | tail -40; fi"
