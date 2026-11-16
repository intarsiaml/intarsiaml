#!/usr/bin/env bash
# mdraft: build the base (main c8367d5a7) and the new binary in one slot: ~/intarsia-mdraft/bin/{base,new}
# fleetq run --host spark2 --load cpu --mem 34 --expect 45 --label "mdraft: build base + new" -- scripts/mdraft/build2.sh
set -euo pipefail
D="$(dirname "${BASH_SOURCE[0]}")"
REV=${BASE_REV:-c8367d5a7} OUTBIN=base "$D/build.sh"
OUTBIN=new "$D/build.sh"
