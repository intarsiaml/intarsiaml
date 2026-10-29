#!/usr/bin/env bash
# Lane gdnx, GPU window 3: nsys profiles (prof.sh: serve --max-batch 16, 16-stream burst, graphs with node trace) of base and dev,
# for per-kernel times at B = 16 in VPROF's units.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 45 --expect 30 --label "gdnx: W3 nsys base vs dev" -- scripts/gdnx/w3.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
date +%T; "$H/prof.sh" base base-b16
date +%T; "$H/prof.sh" dev dev-b16
date +%T
