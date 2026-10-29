#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/chunk4k
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$PWD/.lane/target
cargo test --workspace --release --locked --no-run 2>&1 | tail -4
