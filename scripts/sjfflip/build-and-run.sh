#!/usr/bin/env bash
# sjfflip: rebuild the "new" binary, then the gates, in one slot (the box queue is long, so one wait):
#   fleetq run --host spark1 --quiet gpu,cpu --mem 52 --expect 60 --label "sjfflip build new + gates" -- scripts/sjfflip/build-and-run.sh
set -uo pipefail
D="$(dirname "${BASH_SOURCE[0]}")"
"$D/build.sh" new HEAD || exit 1
"$D/run.sh"
