#!/usr/bin/env bash
# Lane det: CPU-slot build on tatooine (no GPU) in the HIP dev image (after scripts/vat/build.sh). Run ON tatooine:
#   fleetq run --host tatooine --load cpu --expect 30 --label "det build" -- ssh tatooine "STAMP=<stamp> bash -s" < scripts/det/build.sh
# Tree ~/intarsiaml/.worktrees/det, target volume intarsia-cargo-det (seeded from intarsia-cargo-ldf); the release CLI is copied
# to /work/det-intarsia.
set -uo pipefail
export INTARSIA_REPO=$HOME/intarsiaml/.worktrees/det INTARSIA_TARGET_VOL=intarsia-cargo-det
mkdir -p "$INTARSIA_REPO/.det/logs"
D=$INTARSIA_REPO/deploy/tatooine/dev.sh
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
if ! docker volume inspect "$INTARSIA_TARGET_VOL" > /dev/null 2>&1; then
  docker volume create "$INTARSIA_TARGET_VOL" > /dev/null
  docker run --rm --user 0 -v intarsia-cargo-ldf:/a:ro -v "$INTARSIA_TARGET_VOL":/b "$IMG" sh -c 'cp -a /a/. /b/ && chmod 0777 /b' && echo "seeded $INTARSIA_TARGET_VOL"
fi
echo "== build ${STAMP:-} $(date +%T)"
$D env INTARSIA_BUILD_STAMP="${STAMP//_/ }" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > "$INTARSIA_REPO/.det/logs/build.log" 2>&1 < /dev/null
echo "build exit $? $(date +%T)"; grep -E '^error|Finished' "$INTARSIA_REPO/.det/logs/build.log" | tail -5
$D cp /target/release/intarsia /work/det-intarsia < /dev/null
ls -la "$INTARSIA_REPO/det-intarsia"
