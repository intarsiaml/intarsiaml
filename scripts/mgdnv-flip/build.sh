#!/bin/zsh
# mgdnv-flip: build the Metal CLI of origin/main (BASE) and of this worktree. Inside a fleetq slot on endor.
set -e
W=${0:A:h:h:h}; BASE=${BASE:-53384381}
rm -rf $W/target-flip-base/src; mkdir -p $W/target-flip-base/src
git -C $W archive $BASE | tar -x -C $W/target-flip-base/src
( cd $W/target-flip-base/src && CARGO_TARGET_DIR=$W/target-flip-base/t INTARSIA_BUILD_STAMP=flip-base cargo build --release -p intarsia-cli --features metal && cp $W/target-flip-base/t/release/intarsia $W/target-flip-base/intarsia-base )
( cd $W && CARGO_TARGET_DIR=$W/target-flip INTARSIA_BUILD_STAMP=flip-new cargo build --release -p intarsia-cli --features metal && cp $W/target-flip/release/intarsia $W/target-flip/intarsia-new )
ls -l $W/target-flip-base/intarsia-base $W/target-flip/intarsia-new
