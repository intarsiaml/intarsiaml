#!/bin/zsh
# Lane mchunkx: build A (this worktree before any code change = origin/main 9562cc2ee) with --features metal and copy the binary aside
# (a later `cargo test --workspace --release` rebuilds target/release/intarsia WITHOUT the metal feature). Run under fleetq --load cpu.
set -u
W=/opt/code/intarsiaml/.worktrees/mchunkx; cd $W
git diff --quiet HEAD -- crates kernels src || { echo "tree is not main: refusing"; exit 2; }
cargo build --release -p intarsia-cli --features metal 2>&1 | tail -3
mkdir -p .base/bin-main && cp target/release/intarsia .base/bin-main/intarsia || exit 1
echo "## build done $(git rev-parse --short HEAD) $(md5 -q .base/bin-main/intarsia)"
