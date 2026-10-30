#!/bin/bash
# MFP8 build: the metal CLI (served runs) and the batch_metal_real harness binary, target dir .lane/target inside the worktree.
cd /opt/code/intarsiaml/.worktrees/mfp8
# Xcode 27.2's Metal toolchain is not installed: .lane/bin/xcrun (copy in raw/scripts/xcrun-shim) routes metal/metallib to the mounted 27.1 one
export PATH=$PWD/.lane/bin:$PATH
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$PWD/.lane/target
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -3
cp .lane/target/release/intarsia .lane/intarsia-metal
cargo test --release --locked -p intarsia-models --test batch_metal_real --no-run 2>&1 | tail -3
cargo test --release --locked -p intarsia-backend-metal --test rows_exact --no-run 2>&1 | tail -3
