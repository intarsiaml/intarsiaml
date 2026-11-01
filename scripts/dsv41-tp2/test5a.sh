#!/usr/bin/env bash
# Stage 5a: the few-layer runtime on ONE Spark (CUDA) against the official layer outputs (layers 0-2, prefill 16 + 2
# decode steps). Runs on spark2 (l3-r0of1 cache + fixtures copied there); vLLM stays up (~20 GB):
#   fleetq run --host spark2 --quiet gpu,cpu --mem 30 --expect 15 --label "dsv41 tp2 5a few-layer CUDA" -- scripts/dsv41-tp2/test5a.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
ssh -n -o BatchMode=yes spark2 "export FLEETQ_JOB=$FLEETQ_JOB; cat ~/intarsia-dsv41-tp2/bin/STAMP; free -g | head -2; \
  INTARSIA_DSV41_BACKEND=cuda INTARSIA_DSV41_CACHE=\$HOME/intarsia-dsv41-tp2/cache/l3-r0of1 INTARSIA_DSV41_FIXTURES=\$HOME/intarsia-dsv41-tp2/fixtures \
  timeout 900 ~/intarsia-dsv41-tp2/bin/dsv41_runtime few_layer_runtime_matches --nocapture --test-threads 1 2>&1 | tail -40; free -g | head -2"
