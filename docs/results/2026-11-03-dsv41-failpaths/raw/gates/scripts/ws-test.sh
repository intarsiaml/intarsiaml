#!/bin/zsh
cd /opt/code/intarsiaml/.worktrees/dsv41-failpaths && git rev-parse HEAD && CARGO_TARGET_DIR=/opt/code/intarsiaml/.worktrees/failpaths-gates/target-ws cargo test --workspace --no-fail-fast 2>&1
echo "cargo exit $?"
