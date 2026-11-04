#!/bin/zsh
cd /opt/code/intarsiaml/.worktrees/m27-hs-bitwise
export CARGO_TARGET_DIR=/opt/code/intarsiaml/.worktrees/m27-hs-bitwise/target-metal
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -30
cp $CARGO_TARGET_DIR/release/intarsia .scratch/intarsia.hsb
