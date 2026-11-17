#!/bin/zsh
# Lane msm192: build this branch (release cli + naxd_bench) and run the routing test. Under fleetq --load cpu.
set -u
W=/opt/code/intarsiaml/.worktrees/msm192; cd $W || exit 1
zsh scripts/msm192/build.sh || exit 1
cargo test --release -p intarsia-backend-metal --test mxfp4_nax small_m_routing -- --nocapture 2>&1 | tail -15
