#!/bin/zsh
# usage: build2.sh  -- builds the metal CLI of main-src (origin/main, the base) and of this branch into .scratch/intarsia.main / .scratch/intarsia.peel
cd /opt/code/intarsiaml/.worktrees/m35-peel-period
export CARGO_TARGET_DIR=$PWD/.scratch/target-metal
(cd .scratch/main-src && cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -3 && cp $CARGO_TARGET_DIR/release/intarsia ../intarsia.main)
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -3
cp $CARGO_TARGET_DIR/release/intarsia .scratch/intarsia.peel
shasum -a 256 .scratch/intarsia.main .scratch/intarsia.peel
