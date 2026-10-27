#!/usr/bin/env bash
# Lane cgv: build the GPU tests (real-model gates and the kernel unit tests) in the dev tree on spark1 (fleetq --load cpu):
#   fleetq run --host spark1 --load cpu --mem 16 --expect 20 --label "cgv: build tests" -- scripts/cgv/tests-build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
ssh -n -o BatchMode=yes spark1 'export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; cd ~/intarsiaml-cgv/trees/${TREE:-dev2} && cargo test --release --no-run -p intarsia-models --test verify_rows_real --test batch_cuda_real 2>&1 | tail -3; cargo test --release --no-run -p intarsia-backend-cuda --test mxfp4_gemv_mr --test mxfp4_mma --test rows_exact --test rows_registry 2>&1 | tail -6'
