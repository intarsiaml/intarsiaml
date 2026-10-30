#!/bin/bash
# usage: build.sh TAG [SRCDIR]  -> builds intarsia-cli (metal) + batch_metal_real in SRCDIR (default: the worktree), copies to .lane/bin-TAG/
set -u
W=/opt/code/intarsiaml/.worktrees/mmoe
S=${2:-$W}; cd $S
# Xcode 27.2's Metal toolchain is not installed: .lane/bin/xcrun (mfp8's shim) routes metal/metallib to the mounted 27.1 one
export PATH=$W/.lane/bin:$PATH
touch crates/intarsia-backend-metal/build.rs
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$W/.lane/target
T=$1; mkdir -p $W/.lane/bin-$T
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -3
cp $W/.lane/target/release/intarsia $W/.lane/bin-$T/intarsia
for t in batch_metal_real rows_exact; do
  p=intarsia-models; [ $t = rows_exact ] && p=intarsia-backend-metal
  cargo test --release --locked -p $p --test $t --no-run 2>&1 | tail -3
  cp $(ls -t $W/.lane/target/release/deps/$t-* | grep -v '\.d$' | head -1) $W/.lane/bin-$T/$t
done
ls -la $W/.lane/bin-$T
