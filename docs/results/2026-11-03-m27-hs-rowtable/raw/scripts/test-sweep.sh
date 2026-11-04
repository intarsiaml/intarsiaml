#!/bin/zsh
cd /opt/code/intarsiaml/.worktrees/m27-hs-rowtable
export CARGO_TARGET_DIR=/opt/code/intarsiaml/.worktrees/m27-hs-rowtable/target-test
HSB_SWEEP=${2:-300} cargo test --release -p intarsia-backend-metal --test hs_bitwise -- --nocapture 2>&1 | tail -25 | tee ${1:?log}
