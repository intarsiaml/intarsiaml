#!/usr/bin/env bash
# Lane cgv window W4 driver (Mac side, inside a held fleetq slot: FLEETQ_JOB=<id> scripts/cgv/w4.sh [a|b|ab]): the two interleaved A/B runs through le-window.sh.
#   a  the #209 re-check: pre209 (8c562765b, #209's parent) vs main (8f93d2397), MTP off and on, 1k, 256 generated tokens, 5 rounds
#   b  the gate: main vs dev2, MTP off and on, 1k / 4k / 16k, 4 rounds
# (le-window.sh runs each round's arms in order, the next round reversed.)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
D=$(cd "$(dirname "$0")" && pwd); what=${1:-ab}
if [[ $what == *a* ]]; then
  echo "== a: #209 re-check $(date +%T)"
  SIZES=1024 GEN=256 REPS=3 "$D/le-window.sh" a209 5 pre-off:pre209:::--spec_off main-off:main:::--spec_off pre-on:pre209 main-on:main
fi
if [[ $what == *b* ]]; then
  echo "== b: gate $(date +%T)"
  SIZES=1024,4096,16384 GEN=128 REPS=3 "$D/le-window.sh" gate 4 main-off:main:::--spec_off dev-off:dev2:::--spec_off main-on:main dev-on:dev2
fi
echo "== W4 done $(date +%T)"
