#!/usr/bin/env bash
# Lane dh (copy of lane df scripts/df): CPU-slot build on tatooine (no GPU) of one or more trees in the HIP dev image (after scripts/tp2-hip/build.sh). Run ON tatooine:
#   fleetq run --host tatooine --load cpu --expect 25 --label "dh build" -- ssh tatooine "TREES='dev=<stamp>' BIN=dh-intarsia HIPTEST=1 bash -s" < scripts/dh/build.sh
# Tree ~/intarsiaml-dh (the lane's only tree), ONE target volume intarsia-cargo-dh (release CLI and GPU tests; the box has little disk:
# seeded from the shared ldf volume only with >= 30 GB free); the release CLI is copied to ~/intarsiaml-dh/$BIN (default dh-intarsia).
# HIPTEST=1 also compiles (no run) the GPU tests the windows use, in volume intarsia-cargo-dh-<name>-hiptest:
# the HIP crate's tests and intarsia-models' verify_rows_real ($MTESTS adds more --test names).
set -uo pipefail
mkdir -p ~/intarsiaml-dh/.dh/logs
for spec in ${TREES:?TREES=name=stamp ...}; do
  name=${spec%%=*}; stamp=${spec#*=}; stamp=${stamp//_/ }
  export INTARSIA_REPO=$HOME/intarsiaml-dh INTARSIA_TARGET_VOL=intarsia-cargo-dh
  D=$INTARSIA_REPO/deploy/tatooine/dev.sh
  IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
  for v in intarsia-cargo-dh; do  # seed from the shared ldf volume (deps prebuilt), as scripts/hdc/seedbuild.sh, when the disk has room
    if ! docker volume inspect "$v" > /dev/null 2>&1; then
      docker volume create "$v" > /dev/null
      free=$(df -BG --output=avail /var/lib/docker 2>/dev/null | tail -1 | tr -dc 0-9)
      if (( ${free:-0} >= 30 )); then
        docker run --rm --user 0 -v intarsia-cargo-ldf:/a:ro -v "$v":/b "$IMG" sh -c 'cp -a /a/. /b/ && chmod 0777 /b' && echo "seeded $v"
      else
        docker run --rm --user 0 -v "$v":/b "$IMG" chmod 0777 /b; echo "not seeded ($free GB free): cold build"
      fi
    fi
  done
  echo "== $name: $stamp $(date +%T)"
  $D env INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > ~/intarsiaml-dh/.dh/logs/build-${BIN:-dh-intarsia}.log 2>&1 < /dev/null
  echo "build $name exit $? $(date +%T)"; grep -E "^(error|warning: unused)|Finished" ~/intarsiaml-dh/.dh/logs/build-${BIN:-dh-intarsia}.log | tail -5
  $D cp /target/release/intarsia /work/${BIN:-dh-intarsia} < /dev/null
  if [[ "${HIPTEST:-0}" == 1 ]]; then
    $D cargo test --release -p intarsia-backend-hip --no-run > ~/intarsiaml-dh/.dh/logs/hiptest-build-$name.log 2>&1 < /dev/null
    echo "hip tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/intarsiaml-dh/.dh/logs/hiptest-build-$name.log | tail -3
    t=(); for x in verify_rows_real ${MTESTS:-}; do t+=(--test "$x"); done
    $D cargo test --release -p intarsia-models "${t[@]}" --no-run > ~/intarsiaml-dh/.dh/logs/models-build-$name.log 2>&1 < /dev/null
    echo "models gpu tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/intarsiaml-dh/.dh/logs/models-build-$name.log | tail -3
  fi
done
ls -la ~/intarsiaml-dh/dh-intarsia*; df -h / | tail -1
