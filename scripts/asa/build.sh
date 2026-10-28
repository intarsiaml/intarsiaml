#!/usr/bin/env bash
# Lane asa: CPU-slot build on tatooine (no GPU) of one or more trees in the HIP dev image (after scripts/tp2-hip/build.sh). Run ON tatooine:
#   fleetq run --host tatooine --load cpu --expect 25 --label "asa build" -- ssh tatooine "TREES='dev=<stamp>' HIPTEST=1 bash -s" < scripts/asa/build.sh
# Tree ~/intarsiaml-asa-<name>, target volume intarsia-cargo-asa-<name>; the release CLI is copied to /work/asa-intarsia.
# HIPTEST=1 also compiles (no run) the GPU tests the windows use, in volume intarsia-cargo-asa-<name>-hiptest:
# the HIP crate's tests and intarsia-models' verify_rows_real ($MTESTS adds more --test names).
set -uo pipefail
mkdir -p ~/intarsiaml-asa/.asa/logs
for spec in ${TREES:?TREES=name=stamp ...}; do
  name=${spec%%=*}; stamp=${spec#*=}; stamp=${stamp//_/ }
  export INTARSIA_REPO=$HOME/intarsiaml-asa INTARSIA_TARGET_VOL=intarsia-cargo-asa-$name
  D=$INTARSIA_REPO/deploy/tatooine/dev.sh
  IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
  for v in intarsia-cargo-asa-$name intarsia-cargo-asa-$name-hiptest; do  # seed from the shared ldf volume (deps prebuilt), as scripts/hdc/seedbuild.sh
    if ! docker volume inspect "$v" > /dev/null 2>&1; then
      docker volume create "$v" > /dev/null
      docker run --rm --user 0 -v intarsia-cargo-ldf:/a:ro -v "$v":/b "$IMG" sh -c 'cp -a /a/. /b/ && chmod 0777 /b' && echo "seeded $v"
    fi
  done
  echo "== $name: $stamp $(date +%T)"
  $D env INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > ~/intarsiaml-asa/.asa/logs/build-$name.log 2>&1 < /dev/null
  echo "build $name exit $? $(date +%T)"; grep -E '^(error|warning: unused)|Finished' ~/intarsiaml-asa/.asa/logs/build-$name.log | tail -5
  $D cp /target/release/intarsia /work/asa-intarsia < /dev/null
  if [[ "${HIPTEST:-0}" == 1 ]]; then
    export INTARSIA_TARGET_VOL=intarsia-cargo-asa-$name-hiptest
    $D cargo test --release -p intarsia-backend-hip --no-run > ~/intarsiaml-asa/.asa/logs/hiptest-build-$name.log 2>&1 < /dev/null
    echo "hip tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/intarsiaml-asa/.asa/logs/hiptest-build-$name.log | tail -3
    t=(); for x in verify_rows_real ${MTESTS:-}; do t+=(--test "$x"); done
    $D cargo test --release -p intarsia-models "${t[@]}" --no-run > ~/intarsiaml-asa/.asa/logs/models-build-$name.log 2>&1 < /dev/null
    echo "models gpu tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/intarsiaml-asa/.asa/logs/models-build-$name.log | tail -3
  fi
done
ls -la ~/intarsiaml-asa-*/asa-intarsia
