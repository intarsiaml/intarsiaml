#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/mprof
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$PWD/.lane/target
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -3
cp .lane/target/release/intarsia .lane/intarsia-metal
cargo test --release --locked -p intarsia-models --test batch_metal_real --no-run 2>&1 | tail -3
