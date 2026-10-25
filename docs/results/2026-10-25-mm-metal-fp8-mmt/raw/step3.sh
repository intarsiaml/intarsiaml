#!/bin/bash
cd /opt/code/wt-mm
export INTARSIA_QWEN36_DIR=$HOME/models/Qwen3.6-35B-A3B-NVFP4 INTARSIA_REQUIRE_GPU=1
B=/opt/code/wt-mm/.mm/bmr-mmt
R=target/release/deps/rows_exact-31ca8bc2b56c4560
INTARSIA_METAL_FP8_MMT=2 $R --test-threads=1 --exact fp8_rows_are_bitwise_the_one_row_gemv 2>&1 | grep -E "test result|panicked"
run() { echo "== $1"; shift; env "$@" $B --ignored --nocapture --test-threads=1 real_35b_batch_step_real_traffic 2>&1 | grep -E "REALTRAFFIC|panicked|error"; }
for rep in 1 2; do
run "base" X=1
run "mmt>=8" INTARSIA_METAL_FP8_MMT=8
run "mmt>=6" INTARSIA_METAL_FP8_MMT=6
done
