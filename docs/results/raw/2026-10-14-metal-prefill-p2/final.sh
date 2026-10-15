#!/bin/zsh
# P2 final checks on the merged tree (after merging spike/metal). usage: final.sh <outdir>  (inside a fleetq slot)
W=/opt/code/intarsiaml/.claude/worktrees/agent-a6a0312aabf864646
O=$1; mkdir -p $O
for i in 1 2 3; do ioreg -r -d 1 -w0 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tee $O/gpu-idle-before.txt
cd $W
cargo test -p intarsia-backend-metal --release > $O/cargo-test-metal.log 2>&1; echo "cargo test metal exit $?"
cargo build --release -p intarsia-cli --features metal > $O/build-cli.log 2>&1; echo "build cli exit $?"
$W/target/release/intarsia kernels test --backend metal --device 0 --model /Users/josef/models/Qwen3-8B > $O/kernels-test.out 2> $O/kernels-test.err; echo "kernels test exit $?"
grep -E "attn_|test result" $O/kernels-test.out $O/cargo-test-metal.log
