#!/bin/zsh
# mcopylen: build the Metal CLI of the base commit (BASE, default origin/main, from `git archive` into target/base-src) and of this worktree.
# Cargo hashes the path of a workspace member relative to the workspace root, so two trees built into ONE target dir share artifact names and a stale
# one can be reused (the base binary came out as this tree's): the base builds into its own target dir (seeded from the first one's release deps), and
# the sources of this tree are touched before its build. Run inside a fleetq slot on endor:
#   fleetq run --host endor --load cpu --mem 12 --expect 20 --label "mcopylen: build" -- scripts/mcopylen/build.sh
set -e
W=${0:A:h:h:h}; BASE=${BASE:-origin/main}; BT=$W/target/base-tgt
if [ -z "${ONLY_NEW:-}" ]; then
  rm -rf $W/target/base-src; mkdir -p $W/target/base-src
  git -C $W archive $BASE | tar -x -m -C $W/target/base-src
  [ -d $BT/release ] || { mkdir -p $BT; cp -cR $W/target/release $BT/release; }
  ( cd $W/target/base-src && CARGO_TARGET_DIR=$BT INTARSIA_BUILD_STAMP=mcopylen-base cargo build --release -p intarsia-cli --features metal && cp $BT/release/intarsia $W/target/intarsia-base )
fi
find $W/crates -name '*.rs' -exec touch {} +
( cd $W && CARGO_TARGET_DIR=$W/target INTARSIA_BUILD_STAMP=mcopylen-new cargo build --release -p intarsia-cli --features metal && cp $W/target/release/intarsia $W/target/intarsia-new )
ls -l $W/target/intarsia-base $W/target/intarsia-new
