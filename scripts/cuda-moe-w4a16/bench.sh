#!/usr/bin/env bash
# MoE microbenchmark on spark1 with vLLM UP (relative numbers only: the GPU is idle but memory is shared):
#   fleetq run --host spark1 --mem 12 --label "w4a16 bench" -- scripts/cuda-moe-w4a16/bench.sh "<T list>" "<KEY=v KEY=v>" ...
# Each further argument is one environment setting set, run in turn ("" = default). TREE (default w4-dev) is the tree on
# spark1; ROUNDS (default 1) repeats the whole list.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TS=${1:-2048,4096}; shift
HOST=${HOST:-spark1}
TREE=${TREE:-w4-dev}
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
for round in $(seq 1 "${ROUNDS:-1}"); do
  for cfg in "$@"; do
    echo "== round $round: ${cfg:-default}"
    rsh "cd $R/$TREE && BIN=\$(ls -t target/acc/release/deps/moe_nvfp4-* | grep -v '\\.d\$' | head -1) && env $cfg INTARSIA_BENCH_T=$TS INTARSIA_REQUIRE_GPU=1 \$BIN --ignored --nocapture bench_moe_nvfp4_real_shapes 2>&1 | grep -E 'nvfp4_mma|mma_bf16|w4a16|T  '"
  done
done
