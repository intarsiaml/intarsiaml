#!/bin/zsh
W=/opt/code/intarsiaml/.worktrees/flip-metal-mtp; cd $W
export CARGO_TARGET_DIR=$W/.scratch/target
echo "== build merged $(git rev-parse --short HEAD) (+worktree edits) $(date +%T)"
cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3
cp $CARGO_TARGET_DIR/release/intarsia $W/.scratch/intarsia.merged; ls -la $W/.scratch/intarsia.merged
