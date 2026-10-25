#!/usr/bin/env bash
# Lane ac: CPU-slot build on tatooine (no GPU) of ~/intarsiaml-ac in the HIP dev image. Run:
#   fleetq run --host tatooine --expect 30 --load cpu --label "ac: build" -- ssh tatooine "STAMP='head=..._code-tree=..._dirty=no' bash -s" < scripts/ac/build.sh
# Target volume intarsia-cargo-target-ac; the release CLI is copied to ~/intarsiaml-ac/ac-intarsia (/work/ac-intarsia in the image).
set -uo pipefail
stamp=${STAMP:?STAMP}; stamp=${stamp//_/ }
mkdir -p ~/ac-logs
export INTARSIA_REPO=$HOME/intarsiaml-ac INTARSIA_TARGET_VOL=intarsia-cargo-target-ac
D=$INTARSIA_REPO/deploy/tatooine/dev.sh
echo "== ac: $stamp $(date +%T)"
$D env INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > ~/ac-logs/build.log 2>&1 < /dev/null
echo "build exit $? $(date +%T)"; grep -E '^(error|warning: unused)|Finished' ~/ac-logs/build.log | tail -5
$D cp /target/release/intarsia /work/ac-intarsia < /dev/null
ls -la ~/intarsiaml-ac/ac-intarsia
