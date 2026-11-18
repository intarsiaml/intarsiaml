#!/bin/zsh
# herdflip: build the Metal CLI of origin/main (BASE, pinned) and of this worktree. Inside a fleetq slot on endor (--load cpu).
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
set -e
W=${0:A:h:h:h}; BASE=${BASE:-11cb82fe43e2b1b857e09b72f45d28378554e955}
rm -rf $W/target-flip-base/src; mkdir -p $W/target-flip-base/src
git -C $W archive $BASE | tar -x -C $W/target-flip-base/src
( cd $W/target-flip-base/src && CARGO_TARGET_DIR=$W/target-flip-base/t INTARSIA_BUILD_STAMP=herdflip-base cargo build --release -p intarsia-cli --features metal && cp $W/target-flip-base/t/release/intarsia $W/target-flip-base/intarsia-base )
( cd $W && CARGO_TARGET_DIR=$W/target-flip INTARSIA_BUILD_STAMP=herdflip-new cargo build --release -p intarsia-cli --features metal && cp $W/target-flip/release/intarsia $W/target-flip/intarsia-new )
ls -l $W/target-flip-base/intarsia-base $W/target-flip/intarsia-new
