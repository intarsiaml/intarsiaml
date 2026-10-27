#!/usr/bin/env bash
# Lane W4A8, ON tatooine inside a fleetq job (--load cpu): build the release CLI and the hip test binaries in the lane's own
# repo copy and cargo target volume. Sync first (from the Mac): scripts/w4a8/sync.sh
set -uo pipefail
export INTARSIA_REPO=$HOME/intarsiaml-w4a8 INTARSIA_TARGET_VOL=intarsia-cargo-w4a8
cd "$INTARSIA_REPO"
echo "== build release cli --features hip"
deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip 2>&1 | grep -aE "^error|warning: .*hip|Finished" | head -40; echo "rc=${PIPESTATUS[0]}"
echo "== build hip tests (no run)"
deploy/tatooine/dev.sh cargo test --release -p intarsia-backend-hip --no-run 2>&1 | grep -aE "^error|Finished" -A6 | head -60; echo "rc=${PIPESTATUS[0]}"
