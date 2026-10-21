#!/usr/bin/env bash
# Lane G2b: CPU-slot build on tatooine (no GPU) of one or more trees in the HIP dev image. Run ON tatooine:
#   fleetq run --host tatooine --mem 0 --label "g2b build" -- ssh tatooine "TREES='base=head=..._code-tree=..._dirty=no' bash -s" < scripts/35b-tp2-hip/build.sh
# Each tree is ~/intarsiaml-g2b-<name> with its own target volume intarsia-cargo-target-g2b-<name>; the release CLI is copied
# to ~/g2b/bin/<name>-intarsia. `_` in a stamp stands for a space. TEST=1 also runs `cargo test --workspace` (CPU) there.
set -uo pipefail
mkdir -p ~/g2b/bin ~/g2b/logs
for spec in ${TREES:?TREES=name=stamp ...}; do
  name=${spec%%=*}; stamp=${spec#*=}; stamp=${stamp//_/ }
  export INTARSIA_REPO=$HOME/intarsiaml-g2b-$name INTARSIA_TARGET_VOL=intarsia-cargo-target-g2b-$name
  D=$INTARSIA_REPO/deploy/tatooine/dev.sh
  echo "== $name: $stamp $(date +%T)"
  $D env INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > ~/g2b/logs/build-$name.log 2>&1
  echo "build $name exit $? $(date +%T)"; grep -E '^(error|warning: unused)|Finished' ~/g2b/logs/build-$name.log | tail -5
  # the binary needs the image's ROCm libs: it runs inside the image as /work/g2b-intarsia (tree mounted at /work)
  $D cp /target/release/intarsia /work/g2b-intarsia && cp "$INTARSIA_REPO/g2b-intarsia" ~/g2b/bin/$name-intarsia
  if [[ "${TEST:-0}" == 1 ]]; then
    $D cargo test --release --workspace > ~/g2b/logs/test-$name.log 2>&1
    echo "test $name exit $? $(date +%T)"; grep -E '^test result|FAILED|panicked' ~/g2b/logs/test-$name.log | sort | uniq -c | tail -8
  fi
done
ls -la ~/g2b/bin
