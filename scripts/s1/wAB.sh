#!/usr/bin/env bash
# Lane S1 speed window, spark2 side (vLLM down, INSIDE scripts/s1/window.sh): in-engine A/B `intarsia bench speed --at-most`,
# fresh `serve` per arm (scripts/cuda-decode/w-ab.sh via scripts/cuda-prefill4/wB-ab.sh), sizes 1024 / 4096 / 16384, gen 128,
# 2 reps per size, 3 interleaved rounds (base,dev / dev,base / base,dev). base = s1-base (origin/main export), dev = s1-dev.
# AB_ARMS overrides the arm list (e.g. add "devoff:s1-dev:INTARSIA_CUDA_FP8_PAIR_REG=0,INTARSIA_CUDA_NVFP4_REG=0").
# Output ~/intarsia-decfuse/out/s1W/ab
set -u
D=$HOME/intarsia-decfuse/s1-dev/scripts
export BENCH_ARGS="--at-most" TREE=s1-dev OUTNAME=${OUTNAME:-s1W} ROUNDS=${ROUNDS:-3} W_BUDGET_SEC=${W_BUDGET_SEC:-1200}
export ARMS="${AB_ARMS:-base:s1-base dev:s1-dev}"
bash $D/cuda-prefill4/wB-ab.sh
