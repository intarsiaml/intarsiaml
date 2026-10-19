#!/usr/bin/env bash
# One small-GPU slot on spark1 (vLLM UP; relative numbers only): sync + build this working tree as ~/intarsia-decfuse/$TREE
# (default p3-dev) with the CUDA test binaries, then run the steps named in STEPS (comma separated):
#   peak   kernels/cuda_checks/mma_peak.cu (tensor-core throughput, shared-memory limits)
#   gdn    tests gdn_lanes + gdn_timing (lanes vs rows kernel at the 35B shapes)
#   wq     bench_wq_prefill_shapes for each setting in WQ_CFGS (";"-separated "ENV=v ..." sets, "" = default)
#   wqt    the wq kernel tests (WQ_TESTS, default: --test quantmoe)
#   res    register / smem use of the wq kernels
#   fleetq run --host spark1 --mem 12 --label "prefill-3 dev" --wait-timeout 60 -- scripts/cuda-prefill3/dev.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
HOST=${HOST:-spark1}
TREE=${TREE:-p3-dev}
STEPS=${STEPS:-peak,gdn,wq}
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
has() { [[ ",$STEPS," == *",$1,"* ]]; }
if [[ "${NOBUILD:-0}" != 1 ]]; then "$S/../cuda-decode/build.sh" "$TREE" --tests 2>&1 | tail -12; fi
rsh "nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv"
if has peak; then
  echo "== mma_peak"
  rsh "cd $R/$TREE && nvcc -O2 -gencode arch=compute_121,code=sm_121 kernels/cuda_checks/mma_peak.cu -o /tmp/mma_peak_$FLEETQ_JOB && /tmp/mma_peak_$FLEETQ_JOB; rm -f /tmp/mma_peak_$FLEETQ_JOB"
fi
if has gdn; then
  echo "== gdn tests (lanes accuracy, timing)"
  rsh "cd $R/$TREE && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test gdn_lanes --test gdn_timing -- --nocapture --test-threads=1 2>&1 | grep -vE '^\s*(Compiling|Finished|Running)' | tail -60"
fi
if has res; then
  echo "== resources"
  rsh "cd $R/$TREE && d=\$(ls -td target/acc/release/build/intarsia-backend-cuda-*/out | head -1) && for o in fp8_gemm fp8_gemm_f8 nvfp4_gemm ${RES_EXTRA:-}; do echo \"-- \$o\"; cuobjdump --dump-resource-usage \$d/\$o.o 2>&1 | grep -E 'Function|REG' | sed 's/_Z[0-9A-Za-z_]*\\(wq_gemm_kernel\\|gemm_kernel\\|actq\\)/\\1/' | cut -c1-200; done"
fi
if has wqt; then
  echo "== wq tests"
  rsh "cd $R/$TREE && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda ${WQ_TESTS:---test quantmoe} -- --nocapture --test-threads=1 2>&1 | grep -E '^test |rel_l2|differ|panicked|assert|result:' | tail -80"
fi
if has wq; then
  IFS=';' read -r -a cfgs <<< "${WQ_CFGS:-}"
  [[ ${#cfgs[@]} -eq 0 ]] && cfgs=("")
  for round in $(seq 1 "${ROUNDS:-1}"); do
    for cfg in "${cfgs[@]}"; do
      echo "== wq bench round $round: ${cfg:-default}"
      rsh "cd $R/$TREE && BIN=\$(ls -t target/acc/release/deps/quantmoe-* | grep -v '\\.d\$' | head -1) && env $cfg INTARSIA_REQUIRE_GPU=1 \$BIN --ignored --nocapture bench_wq_prefill_shapes 2>&1 | grep -E '^M='"
    done
  done
fi
