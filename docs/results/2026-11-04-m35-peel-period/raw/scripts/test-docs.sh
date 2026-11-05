#!/bin/zsh
cd /opt/code/intarsiaml/.worktrees/m35-peel-period
export CARGO_TARGET_DIR=$PWD/.scratch/target-test
cargo test -p intarsia-cli --test docs_env_vars 2>&1 | tail -8
