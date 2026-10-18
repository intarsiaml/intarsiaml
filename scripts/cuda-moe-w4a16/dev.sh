#!/usr/bin/env bash
# One small-GPU slot (vLLM UP): build tree $TREE (default w4-dev) on spark1, print register / spill / smem of the v2
# kernels, run the v2 tests, then the microbench.
#   fleetq run --host spark1 --mem 12 --label "w4a16 dev (vLLM up)" -- scripts/cuda-moe-w4a16/dev.sh
# Environment: TESTS=0 skips the tests, BENCH="<T list>" (default 1024,2048,4096; empty skips), BENCH_CFG="<ENV=v ..>".
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST=${HOST:-spark1}
TREE=${TREE:-w4-dev}
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
"$S/../cuda-decode/build.sh" "$TREE" --tests 2>&1 | tail -12
echo "== resources"
rsh "cd $R/$TREE && d=\$(ls -td target/acc/release/build/intarsia-backend-cuda-*/out | head -1) && cuobjdump --dump-resource-usage \$d/moe_w4a16.o 2>&1 | grep -E 'Function|REG' | sed 's/_Z[0-9A-Za-z_]*moe_w4a16_kernel/moe_w4a16_kernel/' | cut -c1-200"
if [[ "${TESTS:-1}" == 1 ]]; then
  echo "== tests"
  TREE=$TREE "$S/../cuda-decode/gpu-test.sh" "$TREE" --test moe_w4a16 -- --nocapture --test-threads=1 2>&1 | tail -40
fi
BT=${BENCH-1024,2048,4096}
if [[ -n "$BT" ]]; then
  echo "== microbench"
  "$S/bench.sh" "$BT" "${BENCH_CFG:-}"
fi
