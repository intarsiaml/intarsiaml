#!/usr/bin/env bash
# Lane dcp: CPU-slot build on tatooine (no GPU) of one or more trees in the HIP dev image (after scripts/tp2-hip/build.sh). Run ON tatooine:
#   fleetq run --host tatooine --load cpu --expect 25 --label "dcp build" -- ssh tatooine "TREES='dev=<stamp>' HIPTEST=1 bash -s" < scripts/dcp/build.sh
# Tree ~/intarsiaml-dcp-<name>, target volume intarsia-cargo-target-dcp-<name>; the release CLI is copied to /work/dcp-intarsia.
# HIPTEST=1 also compiles (no run) the GPU tests the windows use, in volume intarsia-cargo-target-dcp-<name>-hiptest:
# the HIP crate's tests and intarsia-models' verify_rows_real ($MTESTS adds more --test names).
set -uo pipefail
mkdir -p ~/intarsiaml-dcp/.dcp/logs
for spec in ${TREES:?TREES=name=stamp ...}; do
  name=${spec%%=*}; stamp=${spec#*=}; stamp=${stamp//_/ }
  export INTARSIA_REPO=$HOME/intarsiaml-dcp INTARSIA_TARGET_VOL=intarsia-cargo-target-dcp-$name
  D=$INTARSIA_REPO/deploy/tatooine/dev.sh
  echo "== $name: $stamp $(date +%T)"
  $D env INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > ~/intarsiaml-dcp/.dcp/logs/build-$name.log 2>&1 < /dev/null
  echo "build $name exit $? $(date +%T)"; grep -E '^(error|warning: unused)|Finished' ~/intarsiaml-dcp/.dcp/logs/build-$name.log | tail -5
  $D cp /target/release/intarsia /work/dcp-intarsia < /dev/null
  if [[ "${HIPTEST:-0}" == 1 ]]; then
    export INTARSIA_TARGET_VOL=intarsia-cargo-target-dcp-$name-hiptest
    $D cargo test --release -p intarsia-backend-hip --no-run > ~/intarsiaml-dcp/.dcp/logs/hiptest-build-$name.log 2>&1 < /dev/null
    echo "hip tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/intarsiaml-dcp/.dcp/logs/hiptest-build-$name.log | tail -3
    t=(); for x in verify_rows_real ${MTESTS:-}; do t+=(--test "$x"); done
    $D cargo test --release -p intarsia-models "${t[@]}" --no-run > ~/intarsiaml-dcp/.dcp/logs/models-build-$name.log 2>&1 < /dev/null
    echo "models gpu tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/intarsiaml-dcp/.dcp/logs/models-build-$name.log | tail -3
  fi
done
ls -la ~/intarsiaml-dcp-*/dcp-intarsia
