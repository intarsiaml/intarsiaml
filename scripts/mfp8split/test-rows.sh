#!/bin/zsh
# Lane mfp8split: unit tests of the plan (lib) and the rows_exact GPU tests (non-ignored). Run under fleetq (default GPU load).
set -u
W=/opt/code/intarsiaml/.worktrees/mfp8split; cd $W
cargo test --release -p intarsia-backend-metal --lib chunk 2>&1 | grep -aE "^(test |running|test result|error)|FAIL|panicked"
cargo test --release -p intarsia-backend-metal --test rows_exact 2>&1 | grep -aE "^(test |running|test result|error)|FAIL|panicked"
