#!/bin/zsh
# P2 kernel tests and microbenchmark. usage: kbench.sh <outdir>   (run inside a fleetq slot)
W=/opt/code/intarsiaml/.claude/worktrees/agent-a6a0312aabf864646
O=$1
mkdir -p $O
for i in 1 2 3; do ioreg -r -d 1 -w0 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tee $O/gpu-idle-before.txt
cd $W
cargo test -p intarsia-backend-metal --release > $O/cargo-test-metal.log 2>&1; echo "cargo test metal exit $?"
cargo test -p intarsia-backend-metal --release --test flash_prefill -- --nocapture > $O/flash-test.log 2>&1; echo "flash tests exit $?"
B=$W/target/release/examples/flash_bench
$B --reps 8 old flash 0 256 1024 2048 3840 7936 15872 > $O/bench-d128-g4.log 2>&1; echo "bench d128 exit $?"
$B --reps 8 flash attn_flash_d128_s4 attn_flash_d128_s8 3840 > $O/bench-d128-variants.log 2>&1
$B --reps 8 --dim 64 old flash 3840 > $O/bench-d64.log 2>&1
$B --reps 8 --dim 256 old flash 3840 > $O/bench-d256.log 2>&1
$B --reps 8 --heads 8 --kv-heads 8 old flash 3840 > $O/bench-g1.log 2>&1
$B --reps 8 --heads 16 --kv-heads 8 old flash 3840 > $O/bench-g2.log 2>&1
$B --reps 8 --heads 64 --kv-heads 8 old flash 3840 > $O/bench-g8.log 2>&1
$B --reps 8 --chunk 64 old flash 3840 > $O/bench-chunk64.log 2>&1
$B --reps 8 --chunk 1024 old flash 3840 > $O/bench-chunk1024.log 2>&1
cat $O/bench-*.log
grep -h "worst error" $O/flash-test.log
