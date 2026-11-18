#!/bin/zsh
# copymetal: build the Metal CLI of this worktree (origin/main after #456). Inside a fleetq slot on endor:
#   fleetq run --host endor --load cpu --mem 12 --expect 25 --label "copymetal: build" -- scripts/copymetal/build.sh
set -e
W=${0:A:h:h:h}
( cd $W && CARGO_TARGET_DIR=$W/target-copymetal INTARSIA_BUILD_STAMP=copymetal cargo build --release -p intarsia-cli --features metal && cp $W/target-copymetal/release/intarsia $W/target-copymetal/intarsia-new )
ls -l $W/target-copymetal/intarsia-new
