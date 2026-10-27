#!/usr/bin/env bash
# Lane TPO, ON tatooine inside a fleetq job (--load cpu): build the release CLI in the lane's own repo copy and cargo target volume.
# usage: build.sh [copy-name]   copies target/release/intarsia to ~/intarsiaml-tpo/<copy-name> (default tpo-intarsia)
# TESTS=1 also builds the hip test binaries (no run).
set -uo pipefail
export INTARSIA_REPO=$HOME/intarsiaml-tpo INTARSIA_TARGET_VOL=intarsia-cargo-tpo
cd "$INTARSIA_REPO"
echo "== build release cli --features hip"
deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip 2>&1 | grep -aE "^error|warning: .*hip|Finished" | head -40; echo "rc=${PIPESTATUS[0]}"
deploy/tatooine/dev.sh sh -c "cp /target/release/intarsia /work/${1:-tpo-intarsia}" && echo "copied to ${1:-tpo-intarsia}"
if [[ "${TESTS:-0}" == 1 ]]; then
  echo "== build hip tests (no run)"
  deploy/tatooine/dev.sh cargo test --release -p intarsia-backend-hip --no-run 2>&1 | grep -aE "^error|Finished" -A6 | head -60; echo "rc=${PIPESTATUS[0]}"
fi
