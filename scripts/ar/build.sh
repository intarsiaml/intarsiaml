#!/usr/bin/env bash
# Lane ars: CPU-slot build on tatooine (no GPU) of a tree in the HIP dev image (after scripts/asp/build.sh). Run from the Mac, see build-ab.sh:
#   STAMP=<stamp> [TREE=<dir>] [TESTS=1] bash -s < scripts/ar/build.sh        (over ssh tatooine)
# Target volume intarsia-cargo-ars (seeded from intarsia-cargo-det, removed when the lane is done); the release CLI is copied to <TREE>/ars-intarsia
# (/work/ars-intarsia in the container). TESTS=1 also builds the HIP comm / ar_norm integration tests (--no-run) for stages-soak.sh `tests` / `rate`.
set -uo pipefail
mkdir -p ~/intarsiaml-ars/.ars/logs
TREE=${TREE:-$HOME/intarsiaml-ars}
export INTARSIA_REPO=$TREE INTARSIA_TARGET_VOL=intarsia-cargo-ars
D=$TREE/deploy/tatooine/dev.sh
LOG=~/intarsiaml-ars/.ars/logs/build-$(basename "$TREE").log
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
if ! docker volume inspect "$INTARSIA_TARGET_VOL" > /dev/null 2>&1; then
  docker volume create "$INTARSIA_TARGET_VOL" > /dev/null
  docker run --rm --user 0 -v intarsia-cargo-det:/a:ro -v "$INTARSIA_TARGET_VOL":/b "$IMG" sh -c 'cp -a /a/. /b/ && chmod 0777 /b' && echo "seeded $INTARSIA_TARGET_VOL"
fi
stamp=${STAMP:-}; stamp=${stamp//_/ }
echo "== ars: $TREE $stamp $(date +%T)"
$D env INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > "$LOG" 2>&1 < /dev/null
echo "build exit $? $(date +%T)"; grep -E '^error|Finished' "$LOG" | tail -5
$D cp /target/release/intarsia /work/ars-intarsia < /dev/null
ls -la "$TREE/ars-intarsia"
if [[ ${TESTS:-0} == 1 ]]; then
  # the HIP integration tests the window runs (stages-soak.sh tests / rate), built here so the GPU window only runs them
  $D cargo test --release -p intarsia-backend-hip --test comm --test ar_norm --no-run > "$LOG.tests" 2>&1 < /dev/null
  echo "tests build exit $?"; grep -E '^error|Finished|Executable' "$LOG.tests" | tail -4
fi
