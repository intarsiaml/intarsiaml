#!/usr/bin/env bash
# Round-6 dev slot on spark1 (vLLM UP; relative numbers only unless the slot is exclusive): optionally sync + build this
# working tree as ~/intarsia-decfuse/$TREE (default r6-dev) with the CUDA test binaries (NOBUILD=1 skips), then run the
# steps named in STEPS (comma separated):
#   res    registers / shared memory / spills of the prefill kernels (cuobjdump of moe_w4a16.o, fp8_gemm.o, gdn.o, attention_flash.o)
#   moe    bench_moe_nvfp4_real_shapes (tests/moe_nvfp4.rs) at MOE_T (default 1024,2048,4096,8192), env sets in MOE_ENVS
#          (";"-separated, "" = default), ROUNDS rounds
#   wq     bench_wq_prefill_shapes (tests/quantmoe.rs) at WQ_M (default 1024,2048,4096) for WQ_KERN (default
#          matmul_fp8_gemm_reg), env sets in WQ_ENVS (";"-separated)
#   gdn    tests/gdn_timing.rs (35B shapes), env sets in GDN_ENVS
#   att    tests/attention.rs attention_throughput, env sets in ATT_ENVS (";"-separated), lines matching ATT_FILTER
#   peak   kernels/cuda_checks/mma_peak.cu and stream_runs.cu (tensor-core and DRAM-run roofs)
#   test   cargo test of the backend test binaries named in TESTS (e.g. "--test moe_w4a16 --test fp8_gemm_reg")
#   unit   the backend's lib tests (registry pins)
#   fleetq run --host spark1 --mem 12 --label "cuda-round6 dev" -- scripts/cuda-round6/dev.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
HOST=${HOST:-spark1}
TREE=${TREE:-r6-dev}
STEPS=${STEPS:-res}
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
has() { [[ ",$STEPS," == *",$1,"* ]]; }
CT="CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda"
OBJ="d=\$(ls -td target/acc/release/build/intarsia-backend-cuda-*/out | head -1)"
if [[ "${NOBUILD:-0}" != 1 ]]; then
  rsh "mkdir -p $R/$TREE; test -d $R/$TREE/target || cp -a $R/r6-base/target $R/$TREE/target"
  "$REPO/scripts/cuda-decode/build.sh" "$TREE" --tests 2>&1 | tail -4
fi
rsh "nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv"
if has res; then
  echo "== resources"
  rsh "cd $R/$TREE && $OBJ && for o in ${RES_OBJS:-moe_w4a16.o fp8_gemm.o gdn.o attention_flash.o}; do echo \"-- \$o\"; cuobjdump --dump-resource-usage \$d/\$o 2>&1 | grep -E 'Function|REG' | paste - - | sed -E 's/_ZN[0-9A-Za-z_]*_cu_[0-9a-f]+//; s/ (STACK|CONSTANT|TEXTURE|SURFACE|SAMPLER)[^ ]*//g' | grep -E \"${RES_FILTER:-.}\" | cut -c1-220; done"
fi
if has moe; then
  IFS=';' read -r -a menvs <<< "${MOE_ENVS:-}"; [[ ${#menvs[@]} -eq 0 ]] && menvs=("")
  for round in $(seq 1 "${ROUNDS:-1}"); do
    for cfg in "${menvs[@]}"; do
      echo "== moe round $round: ${cfg:-default}"
      rsh "cd $R/$TREE && BIN=\$(ls -t target/acc/release/deps/moe_nvfp4-* | grep -v '\\.d\$' | head -1) && env $cfg INTARSIA_X=0 INTARSIA_BENCH_KERNEL=${MOE_KERNELS:-bf16,v2} INTARSIA_BENCH_T=${MOE_T:-1024,2048,4096,8192} INTARSIA_REQUIRE_GPU=1 \$BIN --ignored --nocapture bench_moe_nvfp4_real_shapes 2>&1 | grep -E 'nvfp4_mma|mma_bf16|w4a16|T  |panicked'"
    done
  done
fi
if has wq; then
  IFS=';' read -r -a wenvs <<< "${WQ_ENVS:-INTARSIA_X=0}"
  for round in $(seq 1 "${ROUNDS:-1}"); do
    for cfg in "${wenvs[@]}"; do
      echo "== wq round $round: $cfg"
      rsh "cd $R/$TREE && BIN=\$(ls -t target/acc/release/deps/quantmoe-* | grep -v '\\.d\$' | head -1) && env $cfg INTARSIA_BENCH_M=${WQ_M:-1024,2048,4096} INTARSIA_BENCH_WQ=${WQ_KERN:-matmul_fp8_gemm_reg} INTARSIA_REQUIRE_GPU=1 \$BIN --ignored --nocapture bench_wq_prefill_shapes 2>&1 | grep -E 'M=|panicked'"
    done
  done
fi
if has gdn; then
  IFS=';' read -r -a genvs <<< "${GDN_ENVS:-}"; [[ ${#genvs[@]} -eq 0 ]] && genvs=("")
  for round in $(seq 1 "${ROUNDS:-1}"); do
    for cfg in "${genvs[@]}"; do
      echo "== gdn round $round: ${cfg:-default}"
      rsh "cd $R/$TREE && env $cfg INTARSIA_X=0 $CT --test gdn_timing -- --nocapture --test-threads=1 2>&1 | grep -E 'us/launch|panicked'"
    done
  done
fi
if has att; then
  IFS=';' read -r -a aenvs <<< "${ATT_ENVS:-INTARSIA_X=0}"
  for round in $(seq 1 "${ROUNDS:-1}"); do
    for cfg in "${aenvs[@]}"; do
      echo "== attention_throughput round $round: $cfg"
      rsh "cd $R/$TREE && env $cfg $CT --test attention -- --ignored --nocapture --test-threads=1 attention_throughput 2>&1 | grep -E \"${ATT_FILTER:-prefill|decode|verify|panicked}\" | head -40"
    done
  done
fi
if has peak; then
  echo "== mma_peak, stream_runs"
  rsh "cd $R/$TREE && mkdir -p target/probe && nvcc -O3 -gencode arch=compute_121,code=sm_121 kernels/cuda_checks/mma_peak.cu -o target/probe/mma_peak && target/probe/mma_peak && nvcc -O3 -gencode arch=compute_121,code=sm_121 kernels/cuda_checks/stream_runs.cu -o target/probe/stream_runs && target/probe/stream_runs"
fi
if has test; then
  echo "== tests ${TESTS:?}"
  rsh "cd $R/$TREE && env ${TEST_ENV:-INTARSIA_X=0} $CT ${TESTS} -- --test-threads=1 2>&1 | grep -E '^test |panicked|differ|result:' | tail -40"
fi
if has unit; then
  echo "== lib tests"
  rsh "cd $R/$TREE && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --lib 2>&1 | grep -E 'panicked|result:|FAILED' | tail -8"
fi
