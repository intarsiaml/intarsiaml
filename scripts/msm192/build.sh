#!/bin/zsh
# Lane msm192: build the release CLI (metal) and naxd_bench in this worktree. usage (under fleetq --load cpu): build.sh [<copy-to-dir>]
set -u
W=/opt/code/intarsiaml/.worktrees/msm192; cd $W || exit 1
cargo build --release -p intarsia-cli --features metal || exit 1
cargo build --release -p intarsia-backend-metal --example naxd_bench || exit 1
if [ -n "${1:-}" ]; then mkdir -p $1; cp target/release/intarsia $1/intarsia; cp target/release/examples/naxd_bench $1/naxd_bench; fi
echo "## built $(git rev-parse --short HEAD)"
