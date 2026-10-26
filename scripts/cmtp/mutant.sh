#!/usr/bin/env bash
# Lane cmtp CUDA mutant gate ON spark1: build the clean tree and one mutant (scripts/cmtp/mutate.py) in a copy of src/ (~/intarsiaml-cmtp/mut,
# target-mut; the ids and speed windows' binary target-dev is never rebuilt), print both binaries' sha256 (they must differ), then run the ids
# gate (scripts/cmtp/ids.sh, arm mtp only, off ids from OFF_FROM) with the mutant binary: it must FAIL on at least one prompt.
#   fleetq run --host spark1 --load cpu --expect 20 --label "cmtp: mutant build" -- ssh spark1 'MODE=build bash -s rollback-skip' < scripts/cmtp/mutant.sh
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 15 --label "cmtp: mutant ids" -- ssh spark1 'MODE=ids OFF_FROM=<dir> bash -s rollback-skip short-0 4k-0' < scripts/cmtp/mutant.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cmtp; name=${1:?mutant name}; shift
case ${MODE:?build|ids} in
build)
  mkdir -p $R/mut && rsync -a --delete --exclude target $R/src/ $R/mut/ || exit 1
  (cd $R/mut && CARGO_TARGET_DIR=$R/target-mut CARGO_BUILD_JOBS=12 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -2)
  cp $R/target-mut/release/intarsia $R/mut-clean.bin; echo "clean  sha256 $(sha256sum $R/mut-clean.bin | cut -c1-16)"
  (cd $R/mut && python3 scripts/cmtp/mutate.py . $name) || exit 1
  (cd $R/mut && CARGO_TARGET_DIR=$R/target-mut CARGO_BUILD_JOBS=12 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -2)
  cp $R/target-mut/release/intarsia $R/mut-$name.bin; echo "mutant $name sha256 $(sha256sum $R/mut-$name.bin | cut -c1-16)"
  [ "$(sha256sum < $R/mut-clean.bin)" != "$(sha256sum < $R/mut-$name.bin)" ] && echo "binaries differ" || { echo "NOT A MUTANT: binary identical"; exit 1; } ;;
ids)
  O=$R/out-mut-$name BIN=$R/mut-$name.bin ARMS="off mtp" SRC=$R/src bash $R/src/scripts/cmtp/ids.sh "$@" ;;
esac
