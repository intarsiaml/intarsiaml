#!/usr/bin/env bash
# Core-op fusion phase 2 CUDA no-op (docs/results/2026-10-19-core-op-fusion-p2.md): fusion-noop.sh (vLLM down) or, with "up",
# fusion-noop-up.sh (vLLM up) on A = ~/intarsia-decfuse/fp2-base (origin/main 49973a51 code) and B = fp2-dev (the branch).
#   scripts/cuda-decode/window.sh 16 "core-op-fusion P2 CUDA no-op" '~/intarsia-decfuse/fp2-dev/scripts/cuda-decode/fusion-noop-p2.sh'
#   fleetq run --host spark1 --mem 30 --label "..." -- ssh spark1 "bash ~/intarsia-decfuse/fp2-dev/scripts/cuda-decode/fusion-noop-p2.sh up"
export NOOP_A=fp2-base NOOP_B=fp2-dev
D=$(dirname "$0")
if [ "${1:-}" = up ]; then exec bash "$D/fusion-noop-up.sh"; else exec bash "$D/fusion-noop.sh"; fi
