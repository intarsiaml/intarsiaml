#!/bin/zsh
# usage: build.sh NAME  (builds the metal CLI at the current checkout into .scratch/NAME)
cd /opt/code/intarsiaml/.worktrees/m35-peel
export CARGO_TARGET_DIR=$PWD/.scratch/target-metal
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -3
cp $CARGO_TARGET_DIR/release/intarsia .scratch/${1:?name}
shasum -a 256 .scratch/$1
