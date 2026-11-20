#!/usr/bin/env bash
# samphost profile of the second binary (captured keyed step): nsys n2 and trace window t2, knob on.
#   FLEETQ_OWNER=claude-samphost fleetq run --host spark2 --quiet gpu,cpu --mem 55 --expect 15 --label "samphost profile 2" -- bash scripts/samphost/jobP2.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
SC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
F=INTARSIA_SAMPLED_VERIFY_FAST=1
bash "$SC/nsys.sh" n2 "" "$F" 0.7:4096 0.7:65536
bash "$SC/window.sh" t2 "" "INTARSIA_SPEC_TRACE=1 $F" 0.7:4096,65536:2
