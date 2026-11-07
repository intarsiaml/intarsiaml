#!/usr/bin/env bash
# Lane au1: copy of scripts/dt1 (lane dt1), repointed to ~/intarsiaml-au1, volume intarsia-cargo-au1, CLI au1-intarsia, container au1-serve, output docs/results/2026-11-06-spec-auto-tp1/raw.
# Lane dw (copy of scripts/dh): CPU-slot build on tatooine (no GPU) of one or more trees in the HIP dev image (after scripts/tp2-hip/build.sh). Run ON tatooine:
#   fleetq run --host tatooine --load cpu --expect 25 --label "au1 build" -- ssh tatooine "TREES='dev=<stamp>' BIN=au1-intarsia HTESTS="win_splitkv" bash -s" < scripts/au1/build.sh
# Tree ~/intarsiaml-au1 (the lane's only tree), ONE target volume intarsia-cargo-au1 (release CLI and GPU tests; lane dw reuses lane dh's; the box has little disk:
# seeded from the shared ldf volume only with >= 30 GB free); the release CLI is copied to ~/intarsiaml-au1/$BIN (default au1-intarsia).
# HIPTEST=1 also compiles (no run) the GPU tests the windows use, in volume intarsia-cargo-au1-<name>-hiptest:
# the HIP crate's tests and intarsia-models' verify_rows_real ($MTESTS adds more --test names).
set -uo pipefail
mkdir -p ~/intarsiaml-au1/.au1/logs
for spec in ${TREES:?TREES=name=stamp ...}; do
  name=${spec%%=*}; stamp=${spec#*=}; stamp=${stamp//_/ }
  export INTARSIA_REPO=$HOME/intarsiaml-au1 INTARSIA_TARGET_VOL=intarsia-cargo-au1
  D=$INTARSIA_REPO/deploy/tatooine/dev.sh
  IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
  for v in intarsia-cargo-au1; do  # seed from the shared ldf volume (deps prebuilt), as scripts/hdc/seedbuild.sh, when the disk has room
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
  $D env INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > ~/intarsiaml-au1/.au1/logs/build-${BIN:-au1-intarsia}.log 2>&1 < /dev/null
  rc=$?; echo "build $name exit $rc $(date +%T)"; grep -E "^(error|warning: unused)|Finished" ~/intarsiaml-au1/.au1/logs/build-${BIN:-au1-intarsia}.log | tail -5
  if (( rc != 0 )); then rm -f ~/intarsiaml-au1/${BIN:-au1-intarsia}; echo "build failed: no ${BIN:-au1-intarsia}"; exit 5; fi
  $D cp /target/release/intarsia /work/${BIN:-au1-intarsia} < /dev/null
  if [[ -n "${HTESTS:-}" ]]; then  # lane dw: compile (no run) only the named GPU test binaries of the HIP crate (HTESTS="win_splitkv attention")
    t=(); for x in $HTESTS; do t+=(--test "$x"); done
    $D cargo test --release -p intarsia-backend-hip "${t[@]}" --no-run > ~/intarsiaml-au1/.au1/logs/hiptest-build-$name.log 2>&1 < /dev/null
    echo "hip tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/intarsiaml-au1/.au1/logs/hiptest-build-$name.log | tail -3
  fi
  if [[ -n "${MTESTS:-}" ]]; then  # lane au1: compile (no run) the named intarsia-models GPU test binaries (MTESTS="dflash2_real"), same args as lib.sh mtest
    t=(); for x in $MTESTS; do t+=(--test "$x"); done
    $D cargo test --release -p intarsia-models "${t[@]}" --no-run > ~/intarsiaml-au1/.au1/logs/mtest-build-$name.log 2>&1 < /dev/null
    echo "models tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/intarsiaml-au1/.au1/logs/mtest-build-$name.log | tail -3
  fi
done
ls -la ~/intarsiaml-au1/au1-intarsia*; df -h / | tail -1
