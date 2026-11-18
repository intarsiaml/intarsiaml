#!/bin/zsh
# mgdnvsoak: build the Metal CLI of this worktree (no engine change; = PR #443 head) into target-mgdnvsoak.
set -e
W=${0:A:h:h:h}
cd $W && CARGO_TARGET_DIR=$W/target-mgdnvsoak INTARSIA_BUILD_STAMP=mgdnvsoak cargo build --release -p intarsia-cli --features metal && cp $W/target-mgdnvsoak/release/intarsia $W/target-mgdnvsoak/intarsia-new
ls -l $W/target-mgdnvsoak/intarsia-new
