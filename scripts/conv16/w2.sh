#!/usr/bin/env bash
# Lane conv16, GPU window 2 (one fleetq slot on spark1, quiet on GPU and CPU):
#   1. served-token identity base vs dev (ident.sh: 21 cells, both `serve --max-batch 16`);
#   2. single-stream decode ids: the 1-stream cells of (1) (equal.py), plus the serve A/B's own 1-stream rows;
#   3. the interleaved serve A/B base vs dev, 3 rounds at 1 / 4 / 8 / 16 streams (le-window.sh, step16's settings).
# Stops before the A/B if (1) fails. base = main 5c39315f6 (~/intarsiaml-conv16/bin/base), dev = this branch (bin/dev).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 45 --label "conv16: W2 identity + serve A/B" -- scripts/conv16/w2.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H/../.."
date +%T; "$H/ident.sh" base dev; ok=$?
echo "identity exit $ok"; [[ $ok == 0 ]] || { echo "ABORT: base vs dev served tokens differ"; exit 2; }
date +%T; "$H/le-window.sh" w2 3 base:base::--max-batch_16 dev:dev::--max-batch_16
date +%T
