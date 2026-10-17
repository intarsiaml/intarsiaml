#!/usr/bin/env bash
# MoE microbenchmark sweep on spark1 with vLLM UP (relative numbers only: the GPU is idle but memory is shared):
#   fleetq run --host spark1 --mem 12 --label "nvfp4-mma bench" -- scripts/cuda-moe-nvfp4/bench.sh "<T list>" "<KEY=v KEY=v>" ...
# Each further argument is one environment setting set (e.g. "INTARSIA_NVFP4_STAGES=2 INTARSIA_MOE_BM=32"), run in turn.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TS=${1:-2048,4096}; shift
HOST=${HOST:-spark1}
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
for cfg in "$@"; do
  echo "== ${cfg:-default}"
  rsh "cd $R/nv-dev && BIN=\$(ls -t target/acc/release/deps/moe_nvfp4-* | grep -v '\\.d\$' | head -1) && env $cfg INTARSIA_BENCH_T=$TS INTARSIA_REQUIRE_GPU=1 \$BIN --ignored --nocapture bench_moe_nvfp4_real_shapes 2>&1 | grep -E 'nvfp4_mma|mma_bf16'"
done
