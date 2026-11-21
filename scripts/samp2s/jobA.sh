#!/usr/bin/env bash
# samp2s gate: ABBA windows (none, fast, fast, none) on one binary. Per window: seeded2 (4k / 32k, three cells), then
#   two concurrent sampled streams T 0.7 top_k 20 at 4k / 32k / 64k on stored prefixes (warm, 8 reps), the same at 4k from cold prompts (8 reps), one stream T 0.7 top_k 0 at 4k (8 reps), two greedy streams at 4k (8 reps).
#   FLEETQ_JOB=<held slot> bash scripts/samp2s/jobA.sh [first-window-index]      
set -uo pipefail
: "${FLEETQ_JOB:?run inside a fleetq slot}"
SC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
F=INTARSIA_SAMPLED_VERIFY_FAST=1
W=(a1 - a2 "$F" a3 "$F" a4 -)
for ((i=${1:-0}*2; i<${#W[@]}; i+=2)); do
  bash "$SC/window.sh" "${W[i]}" "" "${W[i+1]}" seeded2:4096,32768 \
    mt:2:0.7:20:4096,32768,65536:8:warm mt:2:0.7:20:4096:8 mt:1:0.7:0:4096:8 mt:2:0:-:4096:8 || echo "window ${W[i]} exit $?"
done
