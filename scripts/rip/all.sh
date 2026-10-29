#!/usr/bin/env bash
# Lane rip: the gates (gates.sh) and the speed A/B (speed.sh) in ONE fleetq job on spark2 (--quiet gpu,cpu):
#   OUT=... fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 150 --label "rip: gates + speed A/B" -- scripts/rip/all.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
D=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:?}
echo "=== gates $(date +%T)"; OUT=$OUT/gates "$D/gates.sh"
echo "=== speed $(date +%T)"; OUT=$OUT/speed "$D/speed.sh"
echo "=== all done $(date +%T)"
