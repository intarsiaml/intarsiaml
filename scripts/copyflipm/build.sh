#!/bin/zsh
# copyflipm: build the Metal CLI of BASE (default: the merge base with origin/main) and of this worktree. Inside a fleetq slot on endor.
set -e
W=${0:A:h:h:h}; BASE=${BASE:-$(git -C $W merge-base HEAD origin/main)}
rm -rf $W/target-cf-base/src; mkdir -p $W/target-cf-base/src
git -C $W archive $BASE | tar -x -C $W/target-cf-base/src
echo "base = $BASE"
( cd $W/target-cf-base/src && CARGO_TARGET_DIR=$W/target-cf-base/t INTARSIA_BUILD_STAMP=cf-base cargo build --release -p intarsia-cli --features metal && cp $W/target-cf-base/t/release/intarsia $W/target-cf-base/intarsia-base )
( cd $W && CARGO_TARGET_DIR=$W/target-cf cargo build --release -p intarsia-cli --features metal 2>&1 | tail -3 )
