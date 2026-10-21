#!/usr/bin/env bash
# Lane H1: CPU-slot build on tatooine (no GPU) of one or more trees in the HIP dev image (after G2b's build.sh). Run ON tatooine:
#   fleetq run --host tatooine --mem 0 --label "h1 build" -- ssh tatooine "TREES='base=head=..._code-tree=..._dirty=no' bash -s" < scripts/h1-tp2-hip/build.sh
# Each tree is ~/intarsiaml-h1-<name> with its own target volume intarsia-cargo-target-h1-<name>; the release CLI is copied
# to /work/h1-intarsia in the tree. `_` in a stamp stands for a space. TEST=1 also runs `cargo test --workspace` (CPU) in a
# SEPARATE target volume (intarsia-cargo-target-h1-<name>-test), so the hip binary is never relinked without its feature.
set -uo pipefail
mkdir -p ~/h1/logs
for spec in ${TREES:?TREES=name=stamp ...}; do
  name=${spec%%=*}; stamp=${spec#*=}; stamp=${stamp//_/ }
  export INTARSIA_REPO=$HOME/intarsiaml-h1-$name INTARSIA_TARGET_VOL=intarsia-cargo-target-h1-$name
  D=$INTARSIA_REPO/deploy/tatooine/dev.sh
  echo "== $name: $stamp $(date +%T)"
  $D env INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > ~/h1/logs/build-$name.log 2>&1 < /dev/null
  echo "build $name exit $? $(date +%T)"; grep -E '^(error|warning: unused)|Finished' ~/h1/logs/build-$name.log | tail -5
  # the binary needs the image's ROCm libs: it runs inside the image as /work/h1-intarsia (tree mounted at /work)
  $D cp /target/release/intarsia /work/h1-intarsia < /dev/null
  if [[ "${TEST:-0}" == 1 ]]; then
    INTARSIA_TARGET_VOL=intarsia-cargo-target-h1-$name-test $D cargo test --release --workspace > ~/h1/logs/test-$name.log 2>&1 < /dev/null
    echo "test $name exit $? $(date +%T)"; grep -E '^test result|FAILED|panicked' ~/h1/logs/test-$name.log | sort | uniq -c | tail -8
  fi
done
ls -la ~/intarsiaml-h1-*/h1-intarsia
