#!/usr/bin/env bash
# One small-GPU slot (vLLM UP) that regenerates every kept-log check of docs/results/2026-10-17-cuda-moe-nvfp4-mma.md:
#   fleetq run --host spark1 --mem 12 --wait-timeout 600 --label "nvfp4-mma final checks (vLLM up)" -- scripts/cuda-moe-nvfp4/final.sh <out-dir>
# Steps (each output file in <out-dir>): build, omma-tile (20 trials), sass (all objects, base vs dev), microbench x3 over
# T = 64..4096 and stage counts, FP4 test output with printed element counts, admission sweep, then the 14-mutant check.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
OUT=${1:?output dir}
mkdir -p "$OUT"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S="$REPO/scripts/cuda-moe-nvfp4"
step() { echo "== $* $(date +%T)"; }
step build
"$REPO/scripts/cuda-decode/build.sh" nv-dev --tests > "$OUT/build.out" 2>&1; echo "build exit $?"
step omma-tile
"$S/probe.sh" > "$OUT/omma-tile.out" 2>&1; echo "probe exit $?"
step sass
"$S/sass.sh" > "$OUT/sass.out" 2>&1; echo "sass exit $?"
step microbench
: > "$OUT/microbench.out"
for r in 1 2 3; do
  echo "### round $r" >> "$OUT/microbench.out"
  "$S/bench.sh" 64,128,256,512,1024,2048,4096 "INTARSIA_NVFP4_STAGES=2" >> "$OUT/microbench.out" 2>&1
done
"$S/bench.sh" 2048,4096 "INTARSIA_BENCH_KERNEL=fp4 INTARSIA_NVFP4_STAGES=3" "INTARSIA_BENCH_KERNEL=fp4 INTARSIA_NVFP4_STAGES=4" "INTARSIA_BENCH_KERNEL=fp4 INTARSIA_NVFP4_STAGES=2" > "$OUT/microbench-stages.out" 2>&1
step tests
"$REPO/scripts/cuda-decode/gpu-test.sh" nv-dev --test moe_nvfp4 -- --nocapture --test-threads=1 > "$OUT/moe_nvfp4-tests.out" 2>&1; echo "moe_nvfp4 exit $?"
"$REPO/scripts/cuda-decode/gpu-test.sh" nv-dev --test admission -- --nocapture --test-threads=1 every_cuda_kernel_matches_the_reference > "$OUT/admission.out" 2>&1; echo "admission exit $?"
step mutcheck
"$S/mutcheck.sh" > "$OUT/mutcheck.out" 2>&1; echo "mutcheck exit $?"
step done
