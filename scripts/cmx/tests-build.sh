#!/usr/bin/env bash
# Lane cmx: build the real-model GPU tests in the dev tree on spark1 (fleetq --load cpu):
#   fleetq run --host spark1 --load cpu --mem 16 --expect 20 --label "cmx: build real-model tests" -- scripts/cmx/tests-build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
ssh -n -o BatchMode=yes spark1 'export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; cd ~/intarsiaml-cmx/trees/dev && cargo test --release --no-run -p intarsia-models --test verify_rows_real --test batch_cuda_real 2>&1 | tail -6'
