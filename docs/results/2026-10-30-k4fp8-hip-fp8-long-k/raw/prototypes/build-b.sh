#!/usr/bin/env bash
# Run on the GPU box: compile the bytes-staging prototypes (scratch/abl copy, -DK4_BYTES=<cap>).
set -uo pipefail
T=$HOME/intarsiaml-tp2-k4fp8; cd "$T"
INC=scratch/abl bash scripts/k4fp8/bench.sh "$T" b4096 -DK4_BYTES=4096 &
INC=scratch/abl bash scripts/k4fp8/bench.sh "$T" b2048 -DK4_BYTES=2048 &
wait
