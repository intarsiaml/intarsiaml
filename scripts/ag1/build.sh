#!/usr/bin/env bash
# Lane ag1: copy of scripts/dt1 (lane dt1), repointed to ~/intarsiaml-ag1, volume intarsia-cargo-ag1, CLI ag1-intarsia, container ag1-serve, output docs/results/2026-11-07-spec-auto-graph/raw.
# Lane dw (copy of scripts/dh): CPU-slot build on tatooine (no GPU) of one or more trees in the HIP dev image (after scripts/tp2-hip/build.sh). Run ON tatooine:
#   fleetq run --host tatooine --load cpu --expect 25 --label "ag1 build" -- ssh tatooine "TREES='dev=<stamp>' BIN=ag1-intarsia HTESTS="win_splitkv" bash -s" < scripts/ag1/build.sh
# Tree ~/intarsiaml-ag1 (the lane's only tree), ONE target volume intarsia-cargo-ag1 (release CLI and GPU tests; lane dw reuses lane dh's; the box has little disk:
# seeded from the shared ldf volume only with >= 30 GB free); the release CLI is copied to ~/intarsiaml-ag1/$BIN (default ag1-intarsia).
# HIPTEST=1 also compiles (no run) the GPU tests the windows use, in volume intarsia-cargo-ag1-<name>-hiptest:
# the HIP crate's tests and intarsia-models' verify_rows_real ($MTESTS adds more --test names).
set -uo pipefail
mkdir -p ~/intarsiaml-ag1/.ag1/logs
for spec in ${TREES:?TREES=name=stamp ...}; do
  name=${spec%%=*}; stamp=${spec#*=}; stamp=${stamp//_/ }
  export INTARSIA_REPO=$HOME/intarsiaml-ag1 INTARSIA_TARGET_VOL=intarsia-cargo-ag1
  D=$INTARSIA_REPO/deploy/tatooine/dev.sh
  IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
  for v in intarsia-cargo-ag1; do  # seed from the shared ldf volume (deps prebuilt), as scripts/hdc/seedbuild.sh, when the disk has room
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
  $D env INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > ~/intarsiaml-ag1/.ag1/logs/build-${BIN:-ag1-intarsia}.log 2>&1 < /dev/null
  rc=$?; echo "build $name exit $rc $(date +%T)"; grep -E "^(error|warning: unused)|Finished" ~/intarsiaml-ag1/.ag1/logs/build-${BIN:-ag1-intarsia}.log | tail -5
  if (( rc != 0 )); then rm -f ~/intarsiaml-ag1/${BIN:-ag1-intarsia}; echo "build failed: no ${BIN:-ag1-intarsia}"; exit 5; fi
  $D cp /target/release/intarsia /work/${BIN:-ag1-intarsia} < /dev/null
  if [[ -n "${HTESTS:-}" ]]; then  # lane dw: compile (no run) only the named GPU test binaries of the HIP crate (HTESTS="win_splitkv attention")
    t=(); for x in $HTESTS; do t+=(--test "$x"); done
    $D cargo test --release -p intarsia-backend-hip "${t[@]}" --no-run > ~/intarsiaml-ag1/.ag1/logs/hiptest-build-$name.log 2>&1 < /dev/null
    echo "hip tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/intarsiaml-ag1/.ag1/logs/hiptest-build-$name.log | tail -3
  fi
  if [[ -n "${MTESTS:-}" ]]; then  # lane ag1: compile (no run) the named intarsia-models GPU test binaries (MTESTS="dflash2_real"), same args as lib.sh mtest
    t=(); for x in $MTESTS; do t+=(--test "$x"); done
    $D cargo test --release -p intarsia-models "${t[@]}" --no-run > ~/intarsiaml-ag1/.ag1/logs/mtest-build-$name.log 2>&1 < /dev/null
    echo "models tests build $name exit $? $(date +%T)"; grep -E '^error|Finished' ~/intarsiaml-ag1/.ag1/logs/mtest-build-$name.log | tail -3
  fi
done
ls -la ~/intarsiaml-ag1/ag1-intarsia*; df -h / | tail -1
