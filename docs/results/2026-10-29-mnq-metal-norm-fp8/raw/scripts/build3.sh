#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/mnq
export CARGO_TARGET_DIR=$PWD/.lane/target
git checkout -q --detach origin/main || exit 1
cargo build --release -p intarsia-cli --features metal 2>&1 | grep -E "^error" -A8
cp .lane/target/release/intarsia .lane/bin/intarsia-main
git rev-parse --short HEAD > .lane/bin/main.rev
git checkout -q lane-mnq/metal-norm-fp8 || exit 1
cargo build --release -p intarsia-cli --features metal 2>&1 | grep -E "^error" -A8
cp .lane/target/release/intarsia .lane/bin/intarsia-dev
git rev-parse --short HEAD > .lane/bin/dev.rev
echo BUILD3-DONE
