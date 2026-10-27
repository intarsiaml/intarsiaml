#!/usr/bin/env bash
# Lane cgv: ON spark1, inside one fleetq quiet slot (tests built by scripts/cgv/tests-build.sh):
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "cgv: real-model gates" -- ssh spark1 'bash -s' < scripts/cgv/gpu-tests.sh
# 0. the CUDA kernel unit tests (mxfp4_gemv_mr: bitwise the one-row GEMV; mxfp4_mma; rows_exact; rows_registry)
# 1. verify rows == single-row decode steps on the 27B (tests/verify_rows_real.rs), after the GEMM prefill
# 2. batched decode rows == solo step (tests/batch_cuda_real.rs, run with the 27B directory), N = 2..16, 130 steps
# 3. kernels test: the matmul_mxfp4 op full sweep, then the whole CUDA quick sweep
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cgv; M=${MODEL:-$HOME/intarsiaml-cgv/models/Qwen3.8-27B-MXFP4}; O=${O:-$R/out/gates}; mkdir -p $O
cd $R/trees/dev
echo "== unit tests $(date +%T)"
INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test mxfp4_gemv_mr --test mxfp4_mma --test rows_exact --test rows_registry > $O/unit.log 2>&1
echo "rc=$?"; grep -aE "^test |test result|panicked|SKIPPED" $O/unit.log | head -40 | cut -c1-200
echo "== verify rows $(date +%T)"
INTARSIA_VERIFY_DIR=$M INTARSIA_VERIFY_ROWS=4 INTARSIA_VERIFY_LENS=${LENS:-17,130,1024,4000} INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-models --test verify_rows_real -- --nocapture > $O/verify-rows.log 2>&1
echo "rc=$?"; grep -aE "^test |test result|bitwise|identical|MISMATCH|panicked" $O/verify-rows.log | head -20 | cut -c1-200
echo "== batched decode == solo $(date +%T)"
INTARSIA_QWEN36_DIR=$M INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-models --test batch_cuda_real real_35b_batched_decode_is_bitwise_the_solo_step -- --nocapture > $O/batched.log 2>&1
echo "rc=$?"; grep -aE "^test |test result|bitwise identical|panicked|batch plan notes" $O/batched.log | head -20 | cut -c1-200
B=$R/trees/dev/target/release/intarsia
echo "== kernels test --op matmul_mxfp4 $(date +%T)"
$B kernels test --backend cuda --op matmul_mxfp4 > $O/ktest-mxfp4.log 2>&1; echo "rc=$?"; tail -6 $O/ktest-mxfp4.log | cut -c1-200
echo "== kernels test --quick $(date +%T)"
$B kernels test --backend cuda --quick > $O/ktest-quick.log 2>&1; echo "rc=$?"; grep -aE "REJECTED|cases in|admission" $O/ktest-quick.log | tail -4 | cut -c1-200
echo "== done $(date +%T)"
