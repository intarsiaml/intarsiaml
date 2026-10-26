#!/usr/bin/env bash
# Lane mpf build ON spark1 (scratch ~/intarsiaml-mpf only): src/ = this worktree (rsync), target-dev, base/ = origin/main before the lane (git archive), target-base.
#   rsync -a --delete --exclude target --exclude .git --exclude tools/.venv ./ spark1:intarsiaml-mpf/src/
#   fleetq run --host spark1 --load cpu --expect 25 --label "mpf: build" -- ssh spark1 'TREES=dev bash -s' < scripts/mpf/spark1.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-mpf
for t in ${TREES:-dev}; do
  src=$R/src; [ $t = base ] && src=$R/base
  (cd $src && CARGO_TARGET_DIR=$R/target-$t CARGO_BUILD_JOBS=${JOBS:-12} cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  echo "build $t done $(date +%T) $(md5sum $R/target-$t/release/intarsia | cut -c1-12)"
done
