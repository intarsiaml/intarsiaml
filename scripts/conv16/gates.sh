#!/usr/bin/env bash
# Lane conv16: the exactness gates on spark1 in one slot (dev binaries; the same env gives the f32 layout with INTARSIA_CUDA_CONV_BF16=0):
#   gdn_conv16 (bf16 state == f32 state, solo + rows), rows_exact, row_tables, prefix_store_restore_is_bitwise_cold_cuda (35B).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 40 --expect 25 --label "conv16: gates" -- scripts/conv16/gates.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
L='~/intarsiaml-conv16'
RAW=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../docs/results/2026-10-29-conv16-bf16-conv-state/raw; mkdir -p "$RAW"
run() { # name, remote command
  echo "== $1 $(date +%T)"
  ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd $L/src/crates/intarsia-backend-cuda && INTARSIA_REQUIRE_GPU=1 $2 2>&1 | grep -aE '^test |test result|panicked|FAIL|conv16|identical|G-B|bytes' | tail -40" | tee "$RAW/gates-$1.txt"
}
run c16 "timeout 200 $L/bin/gdn_conv16-dev --nocapture --test-threads=1"
run rows_exact "timeout 230 $L/bin/rows_exact-dev --test-threads=1"
run row_tables "timeout 230 $L/bin/row_tables-dev --nocapture --test-threads=1"
echo "== prefix_store $(date +%T)"
ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd $L/src/crates/intarsia-models && INTARSIA_TEST_REAL_MODEL=1 INTARSIA_CUDA_TEST_MODEL=$M35 INTARSIA_REQUIRE_GPU=1 timeout 230 $L/bin/prefix_store_cuda-dev prefix_store_restore_is_bitwise_cold_cuda --nocapture --test-threads=1 2>&1 | grep -aE '^test |test result|panicked|FAIL|bit|identical|equal' | tail -30" | tee "$RAW/gates-prefix_store.txt"
echo "== done $(date +%T)"
