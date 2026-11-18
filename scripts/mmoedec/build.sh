#!/bin/zsh
# mmoedec: build the Metal CLI of the base commit (BASE, default origin/main 9eef7a615, from `git archive`, its OWN target dir
# target-mmoedec-base) and of this worktree (target-mmoedec). Run inside a fleetq slot on endor:
#   fleetq run --host endor --load cpu --mem 12 --expect 20 --label "mmoedec: build" -- scripts/mmoedec/build.sh
set -e
W=${0:A:h:h:h}; BASE=${BASE:-9eef7a615}
if [ -z "${ONLY_NEW:-}" ]; then
  rm -rf $W/target-mmoedec-base/src; mkdir -p $W/target-mmoedec-base/src
  git -C $W archive $BASE | tar -x -C $W/target-mmoedec-base/src
  ( cd $W/target-mmoedec-base/src && CARGO_TARGET_DIR=$W/target-mmoedec-base/t INTARSIA_BUILD_STAMP=mmoedec-base cargo build --release -p intarsia-cli --features metal && cp $W/target-mmoedec-base/t/release/intarsia $W/target-mmoedec-base/intarsia-base )
fi
( cd $W && CARGO_TARGET_DIR=$W/target-mmoedec INTARSIA_BUILD_STAMP=mmoedec-new cargo build --release -p intarsia-cli --features metal && cp $W/target-mmoedec/release/intarsia $W/target-mmoedec/intarsia-new )
ls -l $W/target-mmoedec-base/intarsia-base $W/target-mmoedec/intarsia-new
# temporary counter build (INTARSIA: MOECNT=1 prints one line per MoE experts launch: rows, T, kernel), from the patch in the raw dir; the patch is reverted afterwards
if [ -n "${COUNTER:-}" ]; then
  ( cd $W && git apply docs/results/2026-11-18-mmoedec/raw/count.patch && CARGO_TARGET_DIR=$W/target-mmoedec-cnt INTARSIA_BUILD_STAMP=mmoedec-cnt cargo build --release -p intarsia-cli --features metal; git checkout crates/intarsia-backend-metal/src/moe.rs; cp $W/target-mmoedec-cnt/release/intarsia $W/target-mmoedec-cnt/intarsia-cnt )
fi
