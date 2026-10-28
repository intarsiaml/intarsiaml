#!/usr/bin/env bash
# Lane ar: CPU-slot build on tatooine (no GPU) of ~/intarsiaml-ars in the HIP dev image (after scripts/asp/build.sh). Run from the Mac:
#   fleetq run --host tatooine --load cpu --expect 25 --label "ar build" -- ssh tatooine "STAMP=<stamp> bash -s" < scripts/ar/build.sh
# Target volume intarsia-cargo-ars (seeded from intarsia-cargo-det); the release CLI is copied to ~/intarsiaml-ars/ars-intarsia (/work/ars-intarsia).
set -uo pipefail
mkdir -p ~/intarsiaml-ars/.ars/logs
export INTARSIA_REPO=$HOME/intarsiaml-ars INTARSIA_TARGET_VOL=intarsia-cargo-ars
D=$INTARSIA_REPO/deploy/tatooine/dev.sh
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
if ! docker volume inspect "$INTARSIA_TARGET_VOL" > /dev/null 2>&1; then
  docker volume create "$INTARSIA_TARGET_VOL" > /dev/null
  docker run --rm --user 0 -v intarsia-cargo-det:/a:ro -v "$INTARSIA_TARGET_VOL":/b "$IMG" sh -c 'cp -a /a/. /b/ && chmod 0777 /b' && echo "seeded $INTARSIA_TARGET_VOL"
fi
stamp=${STAMP:-}; stamp=${stamp//_/ }
echo "== ar: $stamp $(date +%T)"
$D env INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > ~/intarsiaml-ars/.ars/logs/build.log 2>&1 < /dev/null
echo "build exit $? $(date +%T)"; grep -E '^error|Finished' ~/intarsiaml-ars/.ars/logs/build.log | tail -5
$D cp /target/release/intarsia /work/ars-intarsia < /dev/null
ls -la ~/intarsiaml-ars/ars-intarsia
