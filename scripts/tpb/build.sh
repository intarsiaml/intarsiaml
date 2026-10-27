#!/usr/bin/env bash
# Lane TPB, ON tatooine inside a fleetq job (--load cpu): build the release CLI and the hip test binaries in the lane's own
# repo copy and cargo target volume. Sync first (from the Mac): scripts/tpb/sync.sh
# usage: build.sh [copy-name]   after the build, copy target/release/intarsia to ~/intarsiaml-tpb/<copy-name> (default tpb-intarsia)
set -uo pipefail
export INTARSIA_REPO=$HOME/intarsiaml-tpb INTARSIA_TARGET_VOL=intarsia-cargo-tpb
cd "$INTARSIA_REPO"
echo "== build release cli --features hip"
deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip 2>&1 | grep -aE "^error|warning: .*hip|Finished" | head -40; echo "rc=${PIPESTATUS[0]}"
deploy/tatooine/dev.sh sh -c "cp /target/release/intarsia /work/${1:-tpb-intarsia}" && echo "copied to ${1:-tpb-intarsia}"
[ "${TESTS:-0}" = 1 ] || exit 0
echo "== build hip tests (no run)"
deploy/tatooine/dev.sh cargo test --release -p intarsia-backend-hip --no-run 2>&1 | grep -aE "^error|Finished" -A6 | head -60; echo "rc=${PIPESTATUS[0]}"
