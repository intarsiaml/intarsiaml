#!/bin/zsh
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
R=$W/docs/results/raw/2026-10-15-metal-prefill-p45/sweep
export NAX_M=256,222,2048,2014
for sh in qo gu down; do
  $W/target/release/examples/gemm_nax bench $sh > $R/bench2-$sh.txt 2>&1
done
NAX_M=256,222,2048 $W/target/release/examples/gemm_nax pair > $R/pair2.txt 2>&1
echo done
