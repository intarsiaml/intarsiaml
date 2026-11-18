#!/bin/zsh
# mgdnv27: build the Metal CLI of this worktree (no engine change; = PR #443 head) into target-mgdnv27.
set -e
W=${0:A:h:h:h}
cd $W && CARGO_TARGET_DIR=$W/target-mgdnv27 INTARSIA_BUILD_STAMP=mgdnv27 cargo build --release -p intarsia-cli --features metal && cp $W/target-mgdnv27/release/intarsia $W/target-mgdnv27/intarsia-new
ls -l $W/target-mgdnv27/intarsia-new
