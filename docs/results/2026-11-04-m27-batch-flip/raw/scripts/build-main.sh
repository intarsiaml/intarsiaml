#!/bin/zsh
cd /opt/code/intarsiaml/.worktrees/m27-flip
export CARGO_TARGET_DIR=$PWD/.scratch/target-metal
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -2
cp $CARGO_TARGET_DIR/release/intarsia .scratch/${1:?name}
shasum -a 256 .scratch/$1
