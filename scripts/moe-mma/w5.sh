#!/usr/bin/env bash
# W5 (spark1, vLLM down, inside scripts/cuda-decode/window.sh): 35b-long on the origin/main build (tree moe-mma-base, clean
# stamp) to see whether the harness reproduces the recorded prefill KL and prefix-reuse verdict of
# docs/results/2026-10-14-35b-accuracy.md section 5 on this day and box (W4 tried this on a dirty base tree and was refused).
set -u
S=$HOME/intarsia-decfuse/moe-mma-dev/scripts/moe-mma/w2.sh
STAGES=long TREE=moe-mma-base OUT=w5base BUDGET_MIN=8 bash $S
echo "== w5 done $(date +%T)"
