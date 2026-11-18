#!/bin/zsh
# mlen15: build the Metal CLI of origin/main (BASE, default: origin/main as fetched) and of this worktree. Inside a fleetq slot on endor.
set -e
W=${0:A:h:h:h}; BASE=${BASE:-$(git -C $W rev-parse origin/main)}
rm -rf $W/target-len15-base/src; mkdir -p $W/target-len15-base/src
git -C $W archive $BASE | tar -x -C $W/target-len15-base/src
echo "base = $BASE"
( cd $W/target-len15-base/src && CARGO_TARGET_DIR=$W/target-len15-base/t INTARSIA_BUILD_STAMP=len15-base cargo build --release -p intarsia-cli --features metal && cp $W/target-len15-base/t/release/intarsia $W/target-len15-base/intarsia-base )
( cd $W && CARGO_TARGET_DIR=$W/target-len15 INTARSIA_BUILD_STAMP=len15-new cargo build --release -p intarsia-cli --features metal && cp $W/target-len15/release/intarsia $W/target-len15/intarsia-new )
ls -l $W/target-len15-base/intarsia-base $W/target-len15/intarsia-new
