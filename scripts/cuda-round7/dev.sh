#!/usr/bin/env bash
# Round-7 dev slot on spark1 (vLLM UP: timings are relative numbers unless the slot is exclusive). Optionally syncs and
# builds this working tree as ~/intarsia-decfuse/$TREE (default r7-dev) with the CUDA test binaries (NOBUILD=1 skips),
# then runs the steps named in STEPS (comma separated):
#   test   the e4m3 GEMM tests of tests/quantmoe.rs (accuracy vs bf16 gemm, row invariance) and the selection pins
#          (lib, with and without INTARSIA_CUDA_FP8_GEMM_LT=1)
#   wq     bench_wq_prefill_shapes at WQ_M (default 2048) for WQ_KERN (default gemm_reg,f8,lt), ROUNDS rounds, envs WQ_ENVS
#   quant  the full backend test suite (cargo test -p intarsia-backend-cuda) once with LT on (TEST_ENV)
#   fleetq run --host spark1 --mem 8 --label "..." -- scripts/cuda-round7/dev.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
HOST=${HOST:-spark1}
TREE=${TREE:-r7-dev}
STEPS=${STEPS:-test,wq}
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
has() { [[ ",$STEPS," == *",$1,"* ]]; }
CT="CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda"
if [[ "${NOBUILD:-0}" != 1 ]]; then
  rsh "mkdir -p $R/$TREE; test -d $R/$TREE/target || cp -a $R/${SEED:-r6-dev}/target $R/$TREE/target"
  "$REPO/scripts/cuda-decode/build.sh" "$TREE" --tests 2>&1 | grep -E "^error|error\[|warning: unused|Finished|intarsia$|stamp" | head -30
fi
rsh "nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader; nvcc --version | tail -1; ls /usr/local/cuda/lib64/libcublasLt.so* | head -2"
if has test; then
  echo "== e4m3 GEMM tests"
  rsh "cd $R/$TREE && env ${TENV:-INTARSIA_X=0} INTARSIA_LT_DEBUG=1 $CT --test quantmoe -- --test-threads=1 --nocapture e4m3 matmul_fp8_gemm_lt matmul_fp8_gemm_f8 2>&1 | grep -E '^test |panicked|differ|rel_l2|result:|matmul_fp8_gemm_lt:' | tail -60"
  echo "== selection pins (default, then INTARSIA_CUDA_FP8_GEMM_LT=1)"
  rsh "cd $R/$TREE && $CT --lib fp8_gemm_selection 2>&1 | grep -E '^test |panicked|result:'; INTARSIA_CUDA_FP8_GEMM_LT=1 $CT --lib fp8_gemm_selection 2>&1 | grep -E '^test |panicked|result:'"
fi
if has wq; then
  IFS=';' read -r -a wenvs <<< "${WQ_ENVS:-INTARSIA_X=0}"
  for round in $(seq 1 "${ROUNDS:-1}"); do
    for cfg in "${wenvs[@]}"; do
      echo "== wq round $round: $cfg"
      rsh "cd $R/$TREE && BIN=\$(ls -t target/acc/release/deps/quantmoe-* | grep -v '\\.d\$' | head -1) && env $cfg INTARSIA_BENCH_M=${WQ_M:-2048} INTARSIA_BENCH_WQ=${WQ_KERN:-matmul_fp8_gemm_reg,matmul_fp8_gemm_f8,matmul_fp8_gemm_lt} INTARSIA_REQUIRE_GPU=1 \$BIN --ignored --nocapture bench_wq_prefill_shapes 2>&1 | grep -E 'M=|panicked|gemm_lt:'"
    done
  done
fi
if has quant; then
  echo "== full backend tests with ${TEST_ENV:-INTARSIA_CUDA_FP8_GEMM_LT=1}"
  rsh "cd $R/$TREE && env ${TEST_ENV:-INTARSIA_CUDA_FP8_GEMM_LT=1} $CT -- --test-threads=1 2>&1 | grep -E 'panicked|FAILED|failed|result:' | tail -40"
fi
