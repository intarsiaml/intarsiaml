#!/usr/bin/env bash
# Lane H27P, ON tatooine inside a fleetq job (--load cpu): build the release CLI and the hip test binaries in the lane's own
# repo copy and cargo target volume. Sync first (from the Mac): scripts/h27p/sync.sh
# usage: build.sh [copy-name]   after the build, copy target/release/intarsia to ~/intarsiaml-h27p/<copy-name> (default h27p-intarsia)
set -uo pipefail
export INTARSIA_REPO=$HOME/intarsiaml-h27p INTARSIA_TARGET_VOL=intarsia-cargo-h27p
cd "$INTARSIA_REPO"
echo "== build release cli --features hip"
deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip 2>&1 | grep -aE "^error|warning: .*hip|Finished" | head -40; echo "rc=${PIPESTATUS[0]}"
deploy/tatooine/dev.sh sh -c "cp /target/release/intarsia /work/${1:-h27p-intarsia}" && echo "copied to ${1:-h27p-intarsia}"
echo "== build hip tests (no run)"
deploy/tatooine/dev.sh cargo test --release -p intarsia-backend-hip --no-run 2>&1 | grep -aE "^error|Finished" -A6 | head -60; echo "rc=${PIPESTATUS[0]}"
