#!/usr/bin/env bash
# Lane c27: ship this worktree to spark1 ~/intarsia-c27/<name> (default src) and build it (cuda, release). FROM THE MAC inside a spark1 cpu slot:
#   fleetq run --host spark1 --load cpu --mem 20 --expect 40 --label "c27 build" -- scripts/c27/cbuild.sh [name]
# Layout on spark1: ~/intarsia-c27/<name>, target ~/intarsia-c27/target-<name> (seeded from ~/intarsiaml-m1/target), models ~/intarsia-c27/models, out ~/intarsia-c27/out.
# Build load capped for the 121 GB unified memory: CARGO_BUILD_JOBS=4, MAX_JOBS=4, FLASHINFER_NVCC_THREADS=1.
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; T="$(cd "$S/../.." && pwd)"
N=${1:-src}
SHA=$(git -C "$T" rev-parse --short=9 HEAD)
X=(--exclude target --exclude .git --exclude .worktrees --exclude .scratch --exclude .c27 --exclude golden --exclude 'docs/results/*/raw' --exclude .claude)
ssh -n -o BatchMode=yes spark1 'mkdir -p ~/intarsia-c27/out'
rsync -a --delete "${X[@]}" "$T/" spark1:intarsia-c27/$N/
ssh -n -o BatchMode=yes spark1 "set -e; export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; cd ~/intarsia-c27
test -d target-$N || { test -d target-src && cp -a target-src target-$N || cp -a ~/intarsiaml-m1/target target-$N; }
(cd $N && find . -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch
 INTARSIA_BUILD_STAMP='c27-$N $SHA' CARGO_TARGET_DIR=\$HOME/intarsia-c27/target-$N CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
echo \"build $N done \$(date +%T) \$(md5sum target-$N/release/intarsia | cut -c1-12)\""
