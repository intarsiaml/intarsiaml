#!/usr/bin/env bash
# Build tree $TREE and run per-kernel nsys of the experiment bits INTARSIA_MOE_EXP (1 no dequant, 2 no mma, 4 no loop loads)
# at T=$1 (v2 only): fleetq run ... -- scripts/cuda-moe-w4a16/exp.sh 2048 "0 1 2 3 4 6 7"
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE=${TREE:-w4-dev}
"$S/../cuda-decode/build.sh" "$TREE" --tests 2>&1 | tail -3
args=()
for x in $2; do args+=("INTARSIA_MOE_EXP=$x"); done
"$S/prof.sh" "$1" v2 "${args[@]}"
