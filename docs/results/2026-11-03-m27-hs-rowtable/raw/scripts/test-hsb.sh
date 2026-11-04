#!/bin/zsh
# usage (inside a fleetq --load cpu slot): test-hsb.sh <log>   (cargo test of the bitwise test, own target-test)
cd /opt/code/intarsiaml/.worktrees/m27-hs-rowtable
export CARGO_TARGET_DIR=/opt/code/intarsiaml/.worktrees/m27-hs-rowtable/target-test
cargo test --release -p intarsia-backend-metal --test hs_bitwise -- --nocapture 2>&1 | tail -40 | tee ${1:?log}
