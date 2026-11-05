#!/bin/zsh
# rebuilds only this branch's metal CLI into .scratch/intarsia.peel (mtimes of the copied target dir were newer than the edited sources)
cd /opt/code/intarsiaml/.worktrees/m35-peel-period
export CARGO_TARGET_DIR=$PWD/.scratch/target-metal
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -3
cp $CARGO_TARGET_DIR/release/intarsia .scratch/intarsia.peel
shasum -a 256 .scratch/intarsia.peel; strings .scratch/intarsia.peel | grep -c "Tail peel 8, period 2048"
