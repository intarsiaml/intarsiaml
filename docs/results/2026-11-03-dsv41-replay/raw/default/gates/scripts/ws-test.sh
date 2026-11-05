#!/bin/zsh
cd /opt/code/intarsiaml/.worktrees/replay-gates/new && git rev-parse HEAD && CARGO_TARGET_DIR=/opt/code/intarsiaml/.worktrees/replay-gates/target-ws cargo test --workspace --no-fail-fast 2>&1
echo "cargo exit $?"
