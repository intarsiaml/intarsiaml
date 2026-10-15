#!/bin/zsh
# pick sweep: realistic DRAM-streamed weights (3 GiB pool), 11 candidate tiles, 14 values of M, single (qo, down) and pair (gu, kv) kernels
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
R=$W/docs/results/raw/2026-10-15-metal-prefill-p45/sweep
pgrep -fl zoom.us > $R/zoom3.txt 2>&1
export NAX_M=32,64,128,205,222,256,384,512,768,973,1024,1536,2014,2048
export NAX_ONLY=t64x64w4,t128x64w4,t64x64w2,t32x128w2,t32x64w1,t128x32w2,t64x32w1,t32x32w1,t128x128w8,t256x64w8,t64x128w4
export NAX_POOL_MIB=3072
for sh in qo down; do
  $W/target/release/examples/gemm_nax bench $sh > $R/bench3-$sh.txt 2>&1
done
NAX_PAIR_ONLY=1 $W/target/release/examples/gemm_nax pair > $R/pair3.txt 2>&1
echo done
