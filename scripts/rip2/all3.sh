#!/usr/bin/env bash
# Lane rip2 GPU window 3 (spark2, --quiet gpu,cpu): kern gates, the 8B speed A/B (knob 0 vs default on), then the 35B gates only
# if the window still has room (started before minute S35_BY, default 25).
#   OUT=... fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 45 --label "rip2: window 3" -- scripts/rip2/all3.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
D=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:?}; T0=$(date +%s)
echo "=== kern $(date +%T)"; OUT=$OUT/gates PARTS=kern "$D/gates.sh"
echo "=== speed $(date +%T)"; OUT=$OUT/speed "$D/speed.sh"
if (( ($(date +%s) - T0) / 60 < ${S35_BY:-25} )); then echo "=== s35 $(date +%T)"; OUT=$OUT/gates PARTS=s35 "$D/gates.sh"; else echo "=== s35 skipped (window)"; fi
echo "=== all done $(date +%T)"
