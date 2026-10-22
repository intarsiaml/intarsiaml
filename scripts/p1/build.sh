#!/usr/bin/env bash
# Lane P1: CPU-slot build on tatooine (no GPU): release CLI with hip. fleetq run --host tatooine --mem 0 -- ssh tatooine 'bash ~/intarsiaml-p1/scripts/p1/build.sh'
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-p1}" INTARSIA_TARGET_VOL="${INTARSIA_TARGET_VOL:-intarsia-cargo-target-p1}"
cd "$INTARSIA_REPO"
deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip < /dev/null 2>&1 | grep -E "^(error|warning: unused)|Finished|-->|^\s+\|" | head -60
