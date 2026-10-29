#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/chunk4k
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$PWD/.lane/target
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -2
cp .lane/target/release/intarsia .lane/intarsia-metal
cargo test --release --locked -p intarsia-models --test batch_metal_real --no-run 2>&1 | tail -1
.lane/intarsia-metal plan show --backend metal --model $HOME/Models/Qwen3.6-35B-A3B-NVFP4 --json </dev/null 2>/dev/null | grep -E '"prefill_chunk' 
