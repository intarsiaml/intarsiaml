#!/usr/bin/env bash
# Lane moe16b, GPU window 1 (one fleetq slot): served-token identity base vs dev (ident.sh, 21 cells), batched == solo on the 35B
# (batch-real.sh), the interleaved serve A/B base vs dev, 3 rounds at 1/4/8/16 streams (le-window.sh, the #222 / #224 settings),
# then an nsys profile at 16 streams of base and of dev (prof.sh). Stops before the A/B if identity or batch-real fails.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 90 --label "moe16b: W1 identity + batch-real + A/B + nsys" -- scripts/moe16b/w1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/moe16b/docs/results/2026-10-28-moe16b-step-overlap/raw
cd "$H/../.."
date +%T; "$H/ident.sh" base dev; ok=$?
echo "identity exit $ok"; [[ $ok == 0 ]] || { echo "ABORT: base vs dev served tokens differ"; exit 2; }
date +%T; "$H/batch-real.sh" run | tee "$RAW/batch-real.txt"
grep -aq "test result: ok" "$RAW/batch-real.txt" || { echo "ABORT: batch_cuda_real failed"; exit 3; }
date +%T; "$H/le-window.sh" w1 3 base:base::--max-batch_16 dev:dev::--max-batch_16 moeonly:dev:INTARSIA_SIDE_PROJ=0:--max-batch_16
date +%T; RANK_ARGS=5 WIN="4 11" N=16 "$H/prof.sh" base base-b16
date +%T; RANK_ARGS=5 WIN="4 11" N=16 "$H/prof.sh" dev dev-b16
date +%T
