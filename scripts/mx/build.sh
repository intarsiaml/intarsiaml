#!/usr/bin/env bash
# Lane mx: CPU-slot build on tatooine (no GPU), after T2's build.sh. Run ON tatooine:
#   fleetq run --host tatooine-r9700 --load cpu --mem 0 --expect 30 --label "mx build" -- \
#     ssh tatooine "TREES='dev=<stamp> base=<stamp>' bash -s" < scripts/mx/build.sh
# Tree ~/intarsiaml-mx-<name>, target volume intarsia-cargo-target-mx-<name> (seeded from SEED_FROM on first use), release
# CLI copied to /work/mx-intarsia in the tree. `_` in a stamp stands for a space. HIPTEST=1 also builds the HIP crate's GPU
# tests (volume ...-hiptest, --no-run); MTEST=1 the intarsia-models batch tests (...-mtest, --no-run); TEST=1 runs the CPU workspace tests in volume ...-test.
set -uo pipefail
mkdir -p ~/mx/logs
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
seed() {
  local v=$1
  docker volume inspect "$v" >/dev/null 2>&1 && return 0
  docker volume create "$v" >/dev/null
  docker run --rm -v "${SEED_FROM:-intarsia-cargo-target-t2-head}":/from:ro -v "$v":/to --user 0 "$IMG" bash -c 'cp -a /from/. /to/ && chmod 0777 /to' < /dev/null
  echo "seeded $v $(date +%T)"
}
for spec in ${TREES:?TREES=name=stamp ...}; do
  name=${spec%%=*}; stamp=${spec#*=}; stamp=${stamp//_/ }
  export INTARSIA_REPO=$HOME/intarsiaml-mx-$name INTARSIA_TARGET_VOL=intarsia-cargo-target-mx-$name
  seed "$INTARSIA_TARGET_VOL"
  D=$INTARSIA_REPO/deploy/tatooine/dev.sh
  echo "== $name: $stamp $(date +%T)"
  $D env INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > ~/mx/logs/build-$name.log 2>&1 < /dev/null
  echo "build $name exit $? $(date +%T)"; grep -E '^(error|warning: unused)|Finished' ~/mx/logs/build-$name.log | tail -5
  $D cp /target/release/intarsia /work/mx-intarsia < /dev/null
  if [[ "${HIPTEST:-0}" == 1 ]]; then
    seed "$INTARSIA_TARGET_VOL-hiptest"
    INTARSIA_TARGET_VOL=$INTARSIA_TARGET_VOL-hiptest $D cargo test --release -p intarsia-backend-hip --no-run > ~/mx/logs/hiptest-build-$name.log 2>&1 < /dev/null
    echo "hip tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/mx/logs/hiptest-build-$name.log | tail -3
  fi
  if [[ "${MTEST:-0}" == 1 ]]; then
    seed "$INTARSIA_TARGET_VOL-mtest"
    INTARSIA_TARGET_VOL=$INTARSIA_TARGET_VOL-mtest $D cargo test --release -p intarsia-models --test batch_recurrent --test batch_cuda_real --no-run > ~/mx/logs/mtest-build-$name.log 2>&1 < /dev/null
    echo "models tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/mx/logs/mtest-build-$name.log | tail -3
  fi
  if [[ "${TEST:-0}" == 1 ]]; then
    seed "$INTARSIA_TARGET_VOL-test"
    INTARSIA_TARGET_VOL=$INTARSIA_TARGET_VOL-test $D cargo test --release --workspace --exclude intarsia-backend-hip > ~/mx/logs/test-$name.log 2>&1 < /dev/null
    echo "test $name exit $? $(date +%T)"; grep -E '^test result|FAILED|panicked' ~/mx/logs/test-$name.log | sort | uniq -c | tail -8
  fi
done
ls -la ~/intarsiaml-mx-*/mx-intarsia
