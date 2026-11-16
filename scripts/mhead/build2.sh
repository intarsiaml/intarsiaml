#!/usr/bin/env bash
# mhead: build the base (main 914267d89) and the new binary in one slot: ~/intarsia-mhead/bin/{base,new}
# fleetq run --host spark2 --load cpu --mem 34 --expect 45 --label "mhead: build base + new" -- scripts/mhead/build2.sh
set -euo pipefail
D="$(dirname "${BASH_SOURCE[0]}")"
REV=${BASE_REV:-914267d89} OUTBIN=base "$D/build.sh"
OUTBIN=new "$D/build.sh"
