#!/bin/zsh
# Lane mchunkx: rebuild the formatted rows_exact test binary and run its non-ignored tests (the GPU part). Run under fleetq (default GPU load).
set -u
W=/opt/code/intarsiaml/.worktrees/mchunkx; cd $W
cargo test --release -p intarsia-backend-metal --test rows_exact 2>&1 | grep -aE "^(test |running|test result|error)|FAIL|panicked"
