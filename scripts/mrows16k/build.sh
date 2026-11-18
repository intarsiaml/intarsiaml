#!/bin/zsh
# mrows16k: build the Metal CLI of the base commit (BASE, default origin/main 835796344, from `git archive` into target-base/src, its own CARGO_TARGET_DIR
# target-base so cargo's path-hashed fingerprints cannot clobber it) and of this worktree (target/). Run inside a fleetq slot on endor:
#   fleetq run --host endor --load cpu --mem 12 --expect 25 --label "mrows16k: build" -- scripts/mrows16k/build.sh
set -e
W=${0:A:h:h:h}; BASE=${BASE:-835796344}
if [ -z "${ONLY_NEW:-}" ]; then
  mkdir -p $W/target-base; rm -rf $W/target-base/src; mkdir -p $W/target-base/src
  git -C $W archive $BASE | tar -x -C $W/target-base/src
  ( cd $W/target-base/src && CARGO_TARGET_DIR=$W/target-base/t INTARSIA_BUILD_STAMP=mrows16k-base cargo build --release -p intarsia-cli --features metal && cp $W/target-base/t/release/intarsia $W/target-base/intarsia-base )
fi
if [ -z "${ONLY_BASE:-}" ]; then
( cd $W && CARGO_TARGET_DIR=$W/target INTARSIA_BUILD_STAMP=mrows16k-new cargo build --release -p intarsia-cli --features metal && cp $W/target/release/intarsia $W/target/intarsia-new )
fi
ls -l $W/target-base/intarsia-base $W/target/intarsia-new
