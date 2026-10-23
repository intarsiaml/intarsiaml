#!/bin/zsh
# sp lane: build base (origin/main, /opt/code/wt-sp-base) and head (this worktree) Metal binaries in parallel,
# separate target dirs, inside ONE non-exclusive fleetq slot.
# fleetq run --host endor --mem 40 --label "sp: base/head metal build" -- zsh build.sh <out dir>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside fleetq slot" >&2; exit 2; }
OUT=${1:?out dir}; mkdir -p $OUT
HEADWT=${0:A:h:h:h:h:h}
b() { (cd $1 && CARGO_TARGET_DIR=$1/target-metal cargo build --release -p intarsia-cli --features metal > $OUT/build-$2.log 2>&1; echo "exit $?" >> $OUT/build-$2.log) }
b /opt/code/wt-sp-base base & b $HEADWT head & wait
for d in /opt/code/wt-sp-base $HEADWT; do echo "$d $(git -C $d rev-parse HEAD) $(md5 -q $d/target-metal/release/intarsia)"; done | tee $OUT/builds.txt
tail -1 $OUT/build-base.log $OUT/build-head.log
