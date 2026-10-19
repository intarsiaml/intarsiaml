#!/usr/bin/env bash
# CUDA no-op window for core-op fusion phase 1 (docs/results/2026-10-19-core-op-fusion-p1.md). Runs ON spark1 inside
# scripts/cuda-decode/window.sh (vLLM down):
#   scripts/cuda-decode/window.sh 16 "core-op-fusion P1 CUDA no-op" '~/intarsia-decfuse/fusion-dev/scripts/cuda-decode/fusion-noop.sh'
# A = ~/intarsia-decfuse/fusion-base (origin/main code), B = fusion-dev (the branch). CUDA registers none of the new ops, so B
# must be A: w-noop.sh lanes without the speed rounds (35b-short golden on both + self-golden cmp, 256 greedy ids of the
# 35B and the 8B, graphs and eager, the two 8B fp32 goldens on B). The plan and admission compares need no vLLM-down window:
# scripts/cuda-decode/fusion-noop-up.sh, in a --mem slot with vLLM up.
set -u
P=$HOME/intarsia-decfuse
D=$(dirname "$0")
A=fusion-base; B=fusion-dev
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
O=$P/out/noop-$B
BA=$P/$A/target/acc/release/intarsia; BB=$P/$B/target/acc/release/intarsia
echo "== fusion-noop start $(date +%T)"
W_BUDGET_MIN=${W_BUDGET_MIN:-9} bash $D/w-noop.sh $A $B nospeed
echo "== fusion-noop done $(date +%T)"
