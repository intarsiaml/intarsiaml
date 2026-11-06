#!/bin/zsh
# builds the branch binary and origin/main's binary (detached worktree under .scratch/base) into .scratch/
set -u
W=/opt/code/intarsiaml/.worktrees/flip-metal-mtp; cd $W
export CARGO_TARGET_DIR=$W/.scratch/target
echo "== build branch $(git rev-parse --short HEAD) $(date +%T)"
cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished|warning: unused' | tail -3
cp $CARGO_TARGET_DIR/release/intarsia $W/.scratch/intarsia.branch
[ -d $W/.scratch/base ] || git worktree add -q --detach $W/.scratch/base origin/main
echo "== build main $(git -C $W/.scratch/base rev-parse --short HEAD) $(date +%T)"
(cd $W/.scratch/base && cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
cp $CARGO_TARGET_DIR/release/intarsia $W/.scratch/intarsia.main
ls -la $W/.scratch/intarsia.*
echo "== done $(date +%T)"
