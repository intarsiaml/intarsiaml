#!/bin/zsh
# Build A (origin/main, git archive in .base/src) with the metal feature into .base/target; copy the binary to .base/bin-main/.
set -e
W=/opt/code/intarsiaml/.worktrees/mchunk4
cd $W/.base/src && CARGO_TARGET_DIR=$W/.base/target cargo build --release -p intarsia-cli --features metal 2>&1 | tail -3
mkdir -p $W/.base/bin-main && cp $W/.base/target/release/intarsia $W/.base/bin-main/intarsia
