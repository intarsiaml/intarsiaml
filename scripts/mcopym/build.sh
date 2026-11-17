#!/bin/zsh
# mcopym: build the Metal CLI of the base commit (BASE, default the lane's branch point 205b1150, from `git archive` into target/base-src) and of this
# worktree, sharing one CARGO_TARGET_DIR (registry deps are compiled once). Run inside a fleetq slot on endor:
#   fleetq run --host endor --load cpu --mem 12 --expect 14 --label "mcopym: build" -- scripts/mcopym/build.sh
set -e
W=${0:A:h:h:h}; export CARGO_TARGET_DIR=$W/target; BASE=${BASE:-205b11506}
if [ -z "${ONLY_NEW:-}" ]; then
  rm -rf $W/target/base-src; mkdir -p $W/target/base-src
  git -C $W archive $BASE | tar -x -C $W/target/base-src
  ( cd $W/target/base-src && INTARSIA_BUILD_STAMP=mcopym-base cargo build --release -p intarsia-cli --features metal && cp $W/target/release/intarsia $W/target/intarsia-base )
fi
( cd $W && INTARSIA_BUILD_STAMP=mcopym-new cargo build --release -p intarsia-cli --features metal && cp $W/target/release/intarsia $W/target/intarsia-new )
ls -l $W/target/intarsia-base $W/target/intarsia-new
