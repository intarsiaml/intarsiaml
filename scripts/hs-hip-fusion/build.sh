#!/usr/bin/env bash
# Lane HS: CPU-slot build on tatooine (no GPU) of one or more trees in the HIP dev image (after G2b's build.sh). Run ON tatooine:
#   fleetq run --host tatooine --mem 0 --label "hs build" -- ssh tatooine "TREES='base=head=..._code-tree=..._dirty=no' bash -s" < scripts/hs-hip-fusion/build.sh
# Each tree is ~/intarsiaml-hs-<name> with its own target volume intarsia-cargo-target-hs-<name>; the release CLI is copied
# to /work/hs-intarsia in the tree. `_` in a stamp stands for a space. TEST=1 also runs `cargo test --workspace` (CPU) in a
# SEPARATE target volume (intarsia-cargo-target-hs-<name>-test), so the hip binary is never relinked without its feature;
# the HIP crate is excluded there (its tests use the GPUs, which need a radiance window).
# HIPTEST=1 compiles the HIP crate's GPU tests (volume intarsia-cargo-target-hs-<name>-hiptest) without running them.
set -uo pipefail
mkdir -p ~/hs/logs
for spec in ${TREES:?TREES=name=stamp ...}; do
  name=${spec%%=*}; stamp=${spec#*=}; stamp=${stamp//_/ }
  export INTARSIA_REPO=$HOME/intarsiaml-hs-$name INTARSIA_TARGET_VOL=intarsia-cargo-target-hs-$name
  D=$INTARSIA_REPO/deploy/tatooine/dev.sh
  echo "== $name: $stamp $(date +%T)"
  $D env INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > ~/hs/logs/build-$name.log 2>&1 < /dev/null
  echo "build $name exit $? $(date +%T)"; grep -E '^(error|warning: unused)|Finished' ~/hs/logs/build-$name.log | tail -5
  # the binary needs the image's ROCm libs: it runs inside the image as /work/hs-intarsia (tree mounted at /work)
  $D cp /target/release/intarsia /work/hs-intarsia < /dev/null
  if [[ "${HIPTEST:-0}" == 1 ]]; then
    # the HIP crate's GPU tests, compiled only (they run in a radiance window with the same volume: no rebuild there)
    INTARSIA_TARGET_VOL=intarsia-cargo-target-hs-$name-hiptest $D cargo test --release -p intarsia-backend-hip --no-run > ~/hs/logs/hiptest-build-$name.log 2>&1 < /dev/null
    echo "hip tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/hs/logs/hiptest-build-$name.log | tail -3
  fi
  if [[ "${TEST:-0}" == 1 ]]; then
    INTARSIA_TARGET_VOL=intarsia-cargo-target-hs-$name-test $D cargo test --release --workspace --exclude intarsia-backend-hip > ~/hs/logs/test-$name.log 2>&1 < /dev/null
    echo "test $name exit $? $(date +%T)"; grep -E '^test result|FAILED|panicked' ~/hs/logs/test-$name.log | sort | uniq -c | tail -8
  fi
done
ls -la ~/intarsiaml-hs-*/hs-intarsia
