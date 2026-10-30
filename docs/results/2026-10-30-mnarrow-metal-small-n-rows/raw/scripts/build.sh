#!/bin/bash
# MNARROW build: usage build.sh TAG. Builds the metal CLI, the batch_metal_real harness and the rows_exact test binary of the
# current worktree state (target dir .lane/target) and copies them to .lane/bins/TAG/ so main and the change can be A/B'd.
# Xcode 27.2's Metal toolchain is not installed: .lane/bin/xcrun (copy in raw/scripts/xcrun-shim) routes metal/metallib to the mounted 27.1 one.
set -u
TAG=${1:?tag}
cd /opt/code/intarsiaml/.worktrees/mnarrow
export PATH=$PWD/.lane/bin:$PATH
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$PWD/.lane/target
mkdir -p .lane/bins/$TAG
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -3
cp .lane/target/release/intarsia .lane/bins/$TAG/intarsia
for t in "intarsia-models batch_metal_real" "intarsia-backend-metal rows_exact"; do
  set -- $t
  cargo test --release --locked -p $1 --test $2 --no-run 2>&1 | tail -3
  cp $(ls -t .lane/target/release/deps/$2-* | grep -v '\.d$' | head -1) .lane/bins/$TAG/$2
done
ls -la .lane/bins/$TAG
