#!/bin/zsh
cd /opt/code/intarsiaml/.worktrees/m27-flip
export CARGO_TARGET_DIR=$PWD/.scratch/target-metal
cargo test --workspace --release --no-fail-fast --locked > .scratch/logs/cargo-test-workspace.log 2>&1
echo "cargo test exit $?"
grep -E "^test result:|FAILED|failed$|panicked" .scratch/logs/cargo-test-workspace.log | awk '/^test result/{p+=$4;f+=$6;i+=$8} !/^test result/{print} END{print p" passed, "f" failed, "i" ignored"}' | cut -c1-200 | tail -12
