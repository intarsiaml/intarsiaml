#!/usr/bin/env bash
# Lane msd2, GPU window 2 (one fleetq slot): served-token identity base vs dev (scripts/msd/ident.sh), batched == solo on the 35B
# (scripts/msd/batch-real.sh), then the interleaved serve A/B, base vs dev, 3 rounds at 1/4/8/16 streams (scripts/msd/le-window.sh,
# the #213 / #222 settings). Stops before the A/B if identity fails.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 80 --label "msd2: W2 identity + batch-real + A/B" -- scripts/msd/w2.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/msd2/docs/results/2026-10-28-msd2-dense-gemv/raw
cd "$H/../.."
date +%T; "$H/ident.sh" base dev; ok=$?
echo "identity exit $ok"; [[ $ok == 0 ]] || { echo "ABORT: base vs dev served tokens differ"; exit 2; }
date +%T; "$H/batch-real.sh" run | tee "$RAW/batch-real.txt"
grep -aq "test result: ok" "$RAW/batch-real.txt" || { echo "ABORT: batch_cuda_real failed"; exit 3; }
date +%T; "$H/le-window.sh" w2 3 base:base::--max-batch_16 dev:dev::--max-batch_16
date +%T
