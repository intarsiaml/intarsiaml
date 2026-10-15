#!/bin/zsh
# Final gates on the merged tree: cargo test (metal, full), kernels test, qualify. One fleetq slot.
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
R=$W/docs/results/raw/2026-10-15-metal-prefill-p45/gates
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R
cd $W
cargo build --release -p intarsia-cli --features metal > $R/build-final.txt 2>&1
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/cargo-test-metal.txt 2>&1; echo "cargo test exit $?"
./target/release/intarsia kernels test --backend metal --device 0 --model $M -v > $R/kernels-test.txt 2>&1; echo "kernels test exit $?"
./target/release/intarsia qualify --model $M --backend metal --bandwidth-gbs 614 --out $R/qualify > $R/qualify-console.log 2>&1; echo "qualify exit $?"
echo done
