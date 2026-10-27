#!/usr/bin/env bash
# Lane W4A8, ON tatooine inside a fleetq job (--load cpu): build the origin/main binary (from ~/intarsiaml-w4a8-base, synced by
# sync-base.sh) and this branch's binary (from ~/intarsiaml-w4a8), both into the lane's cargo target volume, and copy them next to each
# other in ~/intarsiaml-w4a8 as base-intarsia and w4a8-intarsia (g0.sh runs them).
set -uo pipefail
V=intarsia-cargo-w4a8; A=$HOME/intarsiaml-w4a8; B=$HOME/intarsiaml-w4a8-base
echo "== base (origin/main)"
# both trees are mounted at /work: touch every source so cargo (path + mtime fingerprints) rebuilds for the tree that is mounted now
find $B/crates $B/kernels $B/recipes -type f -exec touch {} +
(cd $B && INTARSIA_REPO=$B INTARSIA_TARGET_VOL=$V deploy/tatooine/dev.sh bash -c 'cargo build --release -p intarsia-cli --features hip 2>&1 | grep -aE "^error|Finished"; cp /target/release/intarsia /work/base-intarsia-out')
cp $B/base-intarsia-out $A/base-intarsia && rm -f $B/base-intarsia-out
echo "== branch"
find $A/crates $A/kernels $A/recipes -type f -exec touch {} +
(cd $A && INTARSIA_REPO=$A INTARSIA_TARGET_VOL=$V deploy/tatooine/dev.sh bash -c 'cargo build --release -p intarsia-cli --features hip 2>&1 | grep -aE "^error|Finished"; cp /target/release/intarsia /work/w4a8-intarsia')
ls -l $A/base-intarsia $A/w4a8-intarsia
