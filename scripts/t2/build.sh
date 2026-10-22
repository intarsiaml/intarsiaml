#!/usr/bin/env bash
# Lane T2: CPU-slot build on tatooine (no GPU) of one or more trees in the HIP dev image (after H1's build.sh). Run ON tatooine:
#   fleetq run --host tatooine --mem 0 --label "t2 build" -- ssh tatooine "TREES='head=<stamp> base=<stamp>' bash -s" < scripts/t2/build.sh
# Tree ~/intarsiaml-t2-<name>, target volume intarsia-cargo-target-t2-<name>, release CLI copied to /work/t2-intarsia in the
# tree. `_` in a stamp stands for a space. TEST=1 also runs `cargo test --workspace --exclude intarsia-backend-hip` (CPU) in a
# SEPARATE volume (…-test) so the hip binary is never relinked without its feature. HIPTEST=1 compiles the HIP crate's GPU
# tests (volume …-hiptest) without running them (they run in a radiance window only).
set -uo pipefail
mkdir -p ~/t2/logs
for spec in ${TREES:?TREES=name=stamp ...}; do
  name=${spec%%=*}; stamp=${spec#*=}; stamp=${stamp//_/ }
  export INTARSIA_REPO=$HOME/intarsiaml-t2-$name INTARSIA_TARGET_VOL=intarsia-cargo-target-t2-$name
  D=$INTARSIA_REPO/deploy/tatooine/dev.sh
  echo "== $name: $stamp $(date +%T)"
  $D env INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > ~/t2/logs/build-$name.log 2>&1 < /dev/null
  echo "build $name exit $? $(date +%T)"; grep -E '^(error|warning: unused)|Finished' ~/t2/logs/build-$name.log | tail -5
  $D cp /target/release/intarsia /work/t2-intarsia < /dev/null
  if [[ "${HIPTEST:-0}" == 1 ]]; then
    INTARSIA_TARGET_VOL=intarsia-cargo-target-t2-$name-hiptest $D cargo test --release -p intarsia-backend-hip --no-run > ~/t2/logs/hiptest-build-$name.log 2>&1 < /dev/null
    echo "hip tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/t2/logs/hiptest-build-$name.log | tail -3
  fi
  if [[ "${TEST:-0}" == 1 ]]; then
    INTARSIA_TARGET_VOL=intarsia-cargo-target-t2-$name-test $D cargo test --release --workspace --exclude intarsia-backend-hip > ~/t2/logs/test-$name.log 2>&1 < /dev/null
    echo "test $name exit $? $(date +%T)"; grep -E '^test result|FAILED|panicked' ~/t2/logs/test-$name.log | sort | uniq -c | tail -8
  fi
done
ls -la ~/intarsiaml-t2-*/t2-intarsia
