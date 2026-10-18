#!/usr/bin/env bash
# One small-GPU slot (vLLM UP): rebuild w4-dev from this tree, run the v2 tests, microbench T=256..4096 x 3 rounds (v1, FP4, v2),
# and the timing-only layout simulation (tree w4-exp, built from the experiment commit; not part of the shipped kernel).
#   fleetq run --host spark1 --mem 16 --label "w4a16 final microbench" -- scripts/cuda-moe-w4a16/final.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== dev (build, resources, tests)"
BENCH= "$S/dev.sh"
echo "=== microbench, 3 rounds"
ROUNDS=3 "$S/bench.sh" 256,512,1024,1536,2048,3072,4096 ""
echo "=== layout simulation (w4-exp, timing only)"
ROUNDS=3 TREE=w4-exp "$S/bench.sh" 1024,2048,4096 "INTARSIA_BENCH_KERNEL=v2" "INTARSIA_BENCH_KERNEL=v2 INTARSIA_W4_SIM=1"
