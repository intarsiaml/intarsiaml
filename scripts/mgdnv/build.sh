#!/bin/zsh
# mgdnv: build the Metal CLI of the base commit (BASE, default the lane's branch point 835796344, from `git archive`, its OWN target dir
# target-mgdnv-base) and of this worktree (target-mgdnv). Run inside a fleetq slot on endor:
#   fleetq run --host endor --load cpu --mem 12 --expect 20 --label "mgdnv: build" -- scripts/mgdnv/build.sh
set -e
W=${0:A:h:h:h}; BASE=${BASE:-835796344}
if [ -z "${ONLY_NEW:-}" ]; then
  rm -rf $W/target-mgdnv-base/src; mkdir -p $W/target-mgdnv-base/src
  git -C $W archive $BASE | tar -x -C $W/target-mgdnv-base/src
  ( cd $W/target-mgdnv-base/src && CARGO_TARGET_DIR=$W/target-mgdnv-base/t INTARSIA_BUILD_STAMP=mgdnv-base cargo build --release -p intarsia-cli --features metal && cp $W/target-mgdnv-base/t/release/intarsia $W/target-mgdnv-base/intarsia-base )
fi
( cd $W && CARGO_TARGET_DIR=$W/target-mgdnv INTARSIA_BUILD_STAMP=mgdnv-new cargo build --release -p intarsia-cli --features metal && cp $W/target-mgdnv/release/intarsia $W/target-mgdnv/intarsia-new )
ls -l $W/target-mgdnv-base/intarsia-base $W/target-mgdnv/intarsia-new
