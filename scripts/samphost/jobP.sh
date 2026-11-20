#!/usr/bin/env bash
# samphost profile job: nsys attribution knob off (main's path) and on, then INTARSIA_SPEC_TRACE host-timing windows off and on.
#   FLEETQ_OWNER=claude-samphost fleetq run --host spark2 --quiet gpu,cpu --mem 55 --expect 60 --label "samphost profile" -- bash scripts/samphost/jobP.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
SC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
F=INTARSIA_SAMPLED_VERIFY_FAST=1
bash "$SC/nsys.sh" n0 "" - 0.7:4096 0:4096 0.7:65536
bash "$SC/nsys.sh" n1 "" "$F" 0.7:4096 0.7:65536
bash "$SC/window.sh" t0 "" "INTARSIA_SPEC_TRACE=1" 0.7:4096,65536:2
bash "$SC/window.sh" t1 "" "INTARSIA_SPEC_TRACE=1 $F" 0.7:4096,65536:2
