#!/bin/zsh
# usage (inside a fleetq --load cpu slot): build.sh <output binary name under .scratch>
cd /opt/code/intarsiaml/.worktrees/m27-hs-rowtable
export CARGO_TARGET_DIR=/opt/code/intarsiaml/.worktrees/m27-hs-rowtable/target-metal
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -30
cp $CARGO_TARGET_DIR/release/intarsia .scratch/${1:?name}
