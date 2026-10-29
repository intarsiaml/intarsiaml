#!/usr/bin/env bash
# Lane fp8kv, GPU window 1 (one fleetq slot on spark2, quiet on GPU and CPU), dev binaries of this branch in ~/intarsiaml-fp8kv/bin:
#   1. kernel gates: row_tables (bf16 and the new fp8 row-table attention vs the solo launches, bitwise), the fp8 one again on
#      the full grid (INTARSIA_ATTN_ROWS_PERSIST=0), attn_f8kv (fp8 kernels vs bf16 on the upcast cache);
#   2. batch_cuda_real real_35b_batched* on the bf16 default and on the fp8 cache (INTARSIA_TEST_KV_DTYPE=fp8_e4m3);
#   3. prefix_store_cuda warm == cold on the fp8 cache;
#   4. timing_35b_batched_step (B = 8 / 16, graphs, prompt 1024) bf16 / fp8 / fp8 / bf16.
#   fleetq run --host spark2 --quiet gpu,cpu --mem 45 --expect 45 --label "fp8kv: W1 kernel + batch + prefix gates, step timing" -- scripts/fp8kv/w1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=$H/../../docs/results/2026-10-29-fp8kv-opt-in/raw/w1; mkdir -p $RAW
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
L=\~/intarsiaml-fp8kv
rsh() { ssh -n -o BatchMode=yes spark2 "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
echo "commit $(git -C "$H" rev-parse --short HEAD) $(date +%T)"
rsh "cd $L/src/crates/intarsia-backend-cuda
  echo '== row_tables'; INTARSIA_REQUIRE_GPU=1 timeout 400 $L/bin/row_tables-dev --test-threads=1 --nocapture 2>&1 | grep -aoE '^G-B.*|panicked.*|test result.*|^test .*(ok|FAILED)\$'
  echo '== row_tables f8kv, INTARSIA_ATTN_ROWS_PERSIST=0'; INTARSIA_ATTN_ROWS_PERSIST=0 INTARSIA_REQUIRE_GPU=1 timeout 300 $L/bin/row_tables-dev --test-threads=1 --nocapture attn_decode 2>&1 | grep -aoE '^G-B.*|panicked.*|test result.*'
  echo '== attn_f8kv'; INTARSIA_REQUIRE_GPU=1 timeout 300 $L/bin/attn_f8kv-dev --test-threads=1 --nocapture 2>&1 | grep -aE 'panicked|test result|^test |bitwise|identical' | tail -20" 2>&1 | tee $RAW/kernels.txt
date +%T
for kv in bf16 fp8_e4m3; do
  echo "== batch_cuda_real real_35b_batched kv=$kv $(date +%T)"
  rsh "cd $L/src/crates/intarsia-models && INTARSIA_TEST_KV_DTYPE=$kv INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 timeout 900 $L/bin/batch_cuda_real-dev real_35b_batched --test-threads=1 --nocapture 2>&1 | grep -aE '^test |test result|panicked|FAIL|identical|pairs|^kv |row-table path unavailable|run eagerly' | sort | uniq -c | sort -rn | head -30"
done 2>&1 | tee $RAW/batch-real.txt
date +%T
echo "== prefix_store_cuda kv=fp8_e4m3" | tee $RAW/prefix.txt
rsh "cd $L/src/crates/intarsia-models && INTARSIA_TEST_KV_DTYPE=fp8_e4m3 INTARSIA_TEST_REAL_MODEL=1 INTARSIA_CUDA_TEST_MODEL=$M35 timeout 600 $L/bin/prefix_store_cuda-dev prefix_store_restore_is_bitwise_cold_cuda --test-threads=1 --nocapture 2>&1 | grep -aE 'PREFIX-STORE|test result|panicked' | cut -c1-400" 2>&1 | tee -a $RAW/prefix.txt
date +%T
for kv in bf16 fp8_e4m3 fp8_e4m3 bf16; do
  echo "== step kv=$kv $(date +%T)"
  rsh "cd $L/src/crates/intarsia-models && INTARSIA_STEP_PROMPT=1024 INTARSIA_TEST_KV_DTYPE=$kv INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 timeout 230 $L/bin/batch_cuda_real-dev --ignored --nocapture --test-threads=1 --exact timing_35b_batched_step 2>&1 | grep -aE '^STEP|panicked|test result|run eagerly'"
done 2>&1 | tee $RAW/steptime.txt
echo "== done $(date +%T)"
