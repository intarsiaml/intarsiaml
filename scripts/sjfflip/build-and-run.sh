#!/usr/bin/env bash
# sjfflip: build the "main" (origin/main, as last fetched) and "new" (HEAD = origin/main + this branch) binaries, then the gates, in one slot (the box queue is long, so one wait):
#   fleetq run --host spark1 --quiet gpu,cpu --mem 52 --expect 60 --label "sjfflip build new + gates" -- scripts/sjfflip/build-and-run.sh
set -uo pipefail
D="$(dirname "${BASH_SOURCE[0]}")"
"$D/build.sh" main origin/main || exit 1
"$D/build.sh" new HEAD || exit 1
"$D/run.sh"
