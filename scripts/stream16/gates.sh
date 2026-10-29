#!/usr/bin/env bash
# Lane stream16: the exactness gates on spark1 in one slot (dev binaries, default env = the streaming kernels on):
#   rows_exact (the two streaming tests, then the rest), row_tables.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 40 --expect 25 --label "stream16: gates" -- scripts/stream16/gates.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
#M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
L='~/intarsiaml-stream16'
RAW=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../docs/results/2026-10-29-stream16-gemv-stream/raw; mkdir -p "$RAW"
run() { # name, remote command
  echo "== $1 $(date +%T)"
  ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd $L/src/crates/intarsia-backend-cuda && INTARSIA_REQUIRE_GPU=1 $2 2>&1 | grep -aE '^test |test result|panicked|FAIL|moe|identical|G-B|bytes' | tail -40" | tee "$RAW/gates-$1.txt"
}
run rows_exact-new "timeout 230 $L/bin/rows_exact-dev --test-threads=1 qpipe rows_pipe"
run rows_exact "timeout 230 $L/bin/rows_exact-dev --test-threads=1 --skip qpipe --skip rows_pipe"
run row_tables "timeout 230 $L/bin/row_tables-dev --nocapture --test-threads=1"
echo "== done $(date +%T)"
