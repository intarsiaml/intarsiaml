#!/usr/bin/env bash
# samphost gate: ABBA windows (none, fast, fast, none) on one binary; sampled T=0.7 at 4k / 32k / 64k (8 reps per window), greedy (3 reps),
# seeded replies (two seeds, 4k / 32k / 64k, 700 tokens) for the byte-for-byte comparison.
#   FLEETQ_OWNER=claude-samphost fleetq run --host spark2 --quiet gpu,cpu --mem 55 --expect 120 --label "samphost ABBA" -- bash scripts/samphost/jobA.sh [first-window-index]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
SC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
F=INTARSIA_SAMPLED_VERIFY_FAST=1
export SEEDS=11,23
W=(a1 - a2 "$F" a3 "$F" a4 -)
for ((i=${1:-0}*2; i<${#W[@]}; i+=2)); do
  bash "$SC/window.sh" "${W[i]}" "" "${W[i+1]}" seeded:4096,32768,65536 0.7:4096,32768,65536:8 0:4096,32768,65536:3 || echo "window ${W[i]} exit $?"
done
