#!/bin/zsh
# mmoe16: build the Metal CLI of the base commit (BASE, default the lane's branch point 7cfe3d885, from `git archive`, its OWN target dir
# target-mmoe16-base) and of this worktree (target-mmoe16). Run inside a fleetq slot on endor:
#   fleetq run --host endor --load cpu --mem 12 --expect 20 --label "mmoe16: build" -- scripts/mmoe16/build.sh
set -e
W=${0:A:h:h:h}; BASE=${BASE:-7cfe3d885}
if [ -z "${ONLY_NEW:-}" ]; then
  rm -rf $W/target-mmoe16-base/src; mkdir -p $W/target-mmoe16-base/src
  git -C $W archive $BASE | tar -x -C $W/target-mmoe16-base/src
  ( cd $W/target-mmoe16-base/src && CARGO_TARGET_DIR=$W/target-mmoe16-base/t INTARSIA_BUILD_STAMP=mmoe16-base cargo build --release -p intarsia-cli --features metal && cp $W/target-mmoe16-base/t/release/intarsia $W/target-mmoe16-base/intarsia-base )
fi
( cd $W && CARGO_TARGET_DIR=$W/target-mmoe16 INTARSIA_BUILD_STAMP=mmoe16-new cargo build --release -p intarsia-cli --features metal && cp $W/target-mmoe16/release/intarsia $W/target-mmoe16/intarsia-new )
ls -l $W/target-mmoe16-base/intarsia-base $W/target-mmoe16/intarsia-new
