#!/usr/bin/env bash
# Lane l5, LOCAL: sync the branch snapshot <name> and origin/main, then build both on tatooine (CPU slots). usage: engine-prep.sh <name> [dev|base|both]
set -uo pipefail
n=${1:?}; what=${2:-both}
cd "$(git rev-parse --show-toplevel)"
export FLEETQ_OWNER=claude-f31ccf91-l5
fleetq run --host tatooine-r9700-2 --load none --expect 3 --label "l5: sync snapshots $n + base (no GPU)" -- bash -c "scripts/l5/sync.sh $n && scripts/l5/sync-base.sh" 2>&1 | grep -v '^fleetq'
fleetq run --host tatooine-r9700-2 --load cpu --expect 25 --label "l5: build engine $n ($what) + host tests (CPU)" -- bash -c "ssh tatooine 'NAME=$n WHAT=$what bash -s' < scripts/l5/prep.sh" 2>&1 | grep -v '^fleetq'
