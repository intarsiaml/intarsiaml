#!/usr/bin/env bash
# Slot (vLLM UP, small buffers): sync + build the dev tree on spark1, run the FP4 MoE tests, the admission sweep and the
# existing quantmoe tests, optionally the microbenchmark (BENCH=1), report register/spill usage of the new object.
#   fleetq run --host spark1 --mem 12 --label "nvfp4-mma dev2" -- scripts/cuda-moe-nvfp4/dev2.sh [extra test args]
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
"$REPO/scripts/cuda-decode/build.sh" nv-dev --tests 2>&1 | tail -30
echo "== resource usage of the FP4 object"
rsh "cd $R/nv-dev && d=\$(ls -td target/acc/release/build/intarsia-backend-cuda-*/out | head -1) && cuobjdump --dump-resource-usage \$d/moe_nvfp4_mma_sm120a.o 2>&1 | grep -E 'Function|REG' | sed 's/_Z[0-9A-Za-z_]*moe_nvfp4_kernel/moe_nvfp4_kernel/' | cut -c1-200 | head -20; cuobjdump -sass \$d/moe_nvfp4_mma_sm120a.o | grep -c OMMA"
echo "== moe_nvfp4 tests"
"$REPO/scripts/cuda-decode/gpu-test.sh" nv-dev --test moe_nvfp4 -- --nocapture --test-threads=1 2>&1 | tail -60
if [[ "${BENCH:-0}" == 1 ]]; then
  echo "== microbench"
  "$REPO/scripts/cuda-decode/gpu-test.sh" nv-dev --test moe_nvfp4 -- --ignored --nocapture --test-threads=1 2>&1 | tail -30
fi
if [[ "${REGRESS:-0}" == 1 ]]; then
  echo "== admission + quantmoe + fused_decode"
  "$REPO/scripts/cuda-decode/gpu-test.sh" nv-dev --test admission --test quantmoe --lib -- --test-threads=1 2>&1 | tail -30
fi
