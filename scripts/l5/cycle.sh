#!/usr/bin/env bash
# Lane l5, LOCAL: one harness cycle = snapshot sync (no GPU) + build (CPU) + one radiance-down window on card 1 (stages-bench.sh by default).
#   scripts/l5/cycle.sh <snapshot name = window tag> [stages script]
# Each fleetq step is its own slot; the window step takes radiance-rank1 down and restores it in an EXIT trap (scripts/l5/window.sh).
set -uo pipefail
w=${1:?window tag}; stages=${2:-scripts/l5/stages-bench.sh}
cd "$(git rev-parse --show-toplevel)"
export FLEETQ_OWNER=claude-f31ccf91-l5
fleetq run --host tatooine-r9700-2 --load none --expect 3 --label "l5: sync snapshot $w (no GPU)" -- scripts/l5/sync.sh "$w" 2>&1 | grep -v '^fleetq' | tail -2
fleetq run --host tatooine-r9700-2 --load cpu --expect 6 --label "l5: build f8bench $w (CPU)" -- bash -c "ssh tatooine 'NAME=$w bash -s' < scripts/l5/hbuild.sh" 2>&1 | grep -v '^fleetq' | head -40
NAME=$w WIN=$w STAGES=$stages CAP=${CAP:-900} fleetq run --host tatooine-r9700-2 --takes-down radiance-rank1 --quiet gpu,cpu --mem 4 --expect "${EXPECT:-10}" \
  --label "l5: $w window (card 1)" -- scripts/l5/window.sh 2>&1 | grep -v '^fleetq'
