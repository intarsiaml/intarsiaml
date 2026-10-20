#!/usr/bin/env bash
# Round-6 window 1, spark1 side (vLLM down, INSIDE scripts/cuda-decode/window.sh): interleaved speed A/B base vs dev,
# `intarsia bench speed --at-most` against a fresh `serve` per arm (scripts/cuda-prefill4/wB-ab.sh -> scripts/cuda-decode/
# w-ab.sh), sizes 1024 / 4096 / 16384, gen 128 (prefill and decode), 2 reps per size, ROUNDS (3) rounds.
# base = tree r6-base (origin/main 0df54872 as is); dev = tree r6-dev (round-6 head, defaults). Output ~/intarsia-decfuse/out/r6B/ab
set -u
D=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export BENCH_ARGS="--at-most"
echo "== speed A/B $(date +%T)"
TREE=r6-dev OUTNAME=${OUTNAME:-r6B} ARMS="${AB_ARMS:-base:r6-base:INTARSIA_X=0 dev:r6-dev:INTARSIA_X=0}" W_BUDGET_SEC=${W_BUDGET_SEC:-780} \
  bash $D/cuda-prefill4/wB-ab.sh
