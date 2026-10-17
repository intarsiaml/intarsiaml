#!/usr/bin/env bash
# W4 (spark1, vLLM down, inside scripts/cuda-decode/window.sh): the small-T sweep of the committed kernel with its per-case
# lines, then 35b-long on the origin/main build (tree moe-mma-base) to see whether the harness reproduces the recorded
# prefill KL and prefix-reuse verdict of docs/results/2026-10-14-35b-accuracy.md section 5 on this day and box.
set -u
S=$HOME/intarsia-decfuse/moe-mma-dev/scripts/moe-mma/w2.sh
STAGES=sweep OUT=w4 BUDGET_MIN=3 bash $S
STAGES=long TREE=moe-mma-base OUT=w4base BUDGET_MIN=7 bash $S
echo "== w4 done $(date +%T)"
