#!/usr/bin/env bash
# Lane GLR, ON tatooine inside a fleetq job (--load cpu): build the release CLI in the lane's own repo copy and cargo target volume.
# usage: build.sh [copy-name]   copies target/release/intarsia to ~/intarsiaml-glr/<copy-name> (default glr-intarsia)
# TESTS=1 also builds the hip test binaries (no run).
set -uo pipefail
export INTARSIA_REPO=$HOME/intarsiaml-glr INTARSIA_TARGET_VOL=intarsia-cargo-glr
cd "$INTARSIA_REPO"
# rsync keeps the Mac mtimes: a tree switch (base <-> dev) can look older than the last build to cargo. Force the kernel library to rebuild.
touch kernels/hip/*.hip crates/intarsia-backend-hip/build.rs crates/intarsia-backend-hip/src/*.rs
echo "== build release cli --features hip"
deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip 2>&1 | grep -aE "^error|warning: .*hip|Finished" | head -40; echo "rc=${PIPESTATUS[0]}"
deploy/tatooine/dev.sh sh -c "cp /target/release/intarsia /work/${1:-glr-intarsia}" && echo "copied to ${1:-glr-intarsia}"
if [[ "${TESTS:-0}" == 1 ]]; then
  echo "== build hip tests (no run)"
  deploy/tatooine/dev.sh cargo test --release -p intarsia-backend-hip --no-run 2>&1 | grep -aE "^error|Finished" -A6 | head -60; echo "rc=${PIPESTATUS[0]}"
fi
