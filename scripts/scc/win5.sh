#!/usr/bin/env bash
# Lane scc window 5: fp8 rows micro-timing at each outputs-per-warp tile, then the served A/B of base against dev at each tile.
#   fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 35 --label "scc: ..." -- scripts/scc/win5.sh
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
RAW=docs/results/2026-10-27-scc-cuda-35b-concurrency/raw
ssh -n -o BatchMode=yes spark2 'export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH FLEETQ_JOB='$FLEETQ_JOB'; cd ~/intarsiaml-scc/src && INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-backend-cuda --release --test rows_exact -- --ignored --nocapture --test-threads=1 timing 2>&1 | grep -a "^TIME fp8"' > $RAW/micro-fp8-nt.txt 2>&1
scripts/scc/le-window.sh sweep 3 base:base::--max-batch_16 devauto:dev::--max-batch_16 dev1:dev:INTARSIA_CUDA_FP8_ROWS_NT=1:--max-batch_16 dev2:dev:INTARSIA_CUDA_FP8_ROWS_NT=2:--max-batch_16 dev4:dev:INTARSIA_CUDA_FP8_ROWS_NT=4:--max-batch_16
