#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/mgd
export CARGO_TARGET_DIR=$PWD/target
for m in "$@"; do
  cp run/gdn.metal.good kernels/metal/gdn.metal
  python3 run/mutants.py $m || { echo "$m: mutation failed to apply"; continue; }
  cargo build --release -p intarsia-cli --features metal 2>&1 | grep -E "^error" -A5
  cp target/release/intarsia run/mut/intarsia-$m
done
cp run/gdn.metal.good kernels/metal/gdn.metal
