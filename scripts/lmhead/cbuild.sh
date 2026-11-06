#!/usr/bin/env bash
# Lane lmhead-fp8-default, CUDA: ship the head tree (this worktree) and the base tree (.scratch/base = main af6aa5226) to spark1 ~/lmh-cuda and build both.
# Run FROM THE MAC inside a spark1 cpu slot:
#   fleetq run --host spark1 --load cpu --mem 16 --expect 40 --label "..." -- scripts/lmhead/cbuild.sh
# Layout on spark1: ~/lmh-cuda/{src,base}, targets ~/lmh-cuda/target-{head,base} (seeded from ~/intarsiaml-m1/target), models ~/lmh-cuda/models, out ~/lmh-cuda/out.
# Removal at the end of the lane: scripts/lmhead/cclean.sh
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; T="$(cd "$S/../.." && pwd)"
SHA=$(git -C "$T" rev-parse --short=9 HEAD)
X=(--exclude target --exclude .git --exclude .worktrees --exclude .scratch --exclude golden --exclude 'docs/results/*/raw' --exclude .claude)
ssh -n -o BatchMode=yes spark1 'mkdir -p ~/lmh-cuda/out'
rsync -a --delete "${X[@]}" "$T/" spark1:lmh-cuda/src/
rsync -a --delete "${X[@]}" "$T/.scratch/base/" spark1:lmh-cuda/base/
ssh -n -o BatchMode=yes spark1 "set -e; export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; cd ~/lmh-cuda
for t in head base; do test -d target-\$t || cp -a ~/intarsiaml-m1/target target-\$t; done
for t in head base; do src=src; [ \$t = base ] && src=base
  (cd \$src && find . -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch
   INTARSIA_BUILD_STAMP='lmhead-\$t $SHA' CARGO_TARGET_DIR=\$HOME/lmh-cuda/target-\$t CARGO_BUILD_JOBS=12 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  echo \"build \$t done \$(date +%T) \$(md5sum target-\$t/release/intarsia | cut -c1-12)\"
done"
