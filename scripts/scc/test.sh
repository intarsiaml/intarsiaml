#!/usr/bin/env bash
# Lane scc: build + run the CUDA backend tests of the synced tree on spark2 (after scripts/scc/build.sh synced it).
#   fleetq run --host spark2 --load cpu --expect 20 --label "scc: build tests" -- scripts/scc/test.sh build
#   fleetq run --host spark2 --mem 8 --expect 15 --label "scc: rows_exact" -- scripts/scc/test.sh run <test-args...>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark2}; MODE=$1; shift
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-scc/src && if [ $MODE = build ]; then CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test rows_exact --test rows_registry --test quick_coverage --no-run 2>&1 | tail -6; else INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-backend-cuda --release $* 2>&1 | tail -40; fi"
