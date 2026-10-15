#!/bin/zsh
# tile sweep, microbench (examples/gemm_nax bench), all NAX variants, per Qwen3-8B shape, M = 256, 222, 1024, 2048
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
R=$W/docs/results/raw/2026-10-15-metal-prefill-p45/sweep
pgrep -fl zoom.us > $R/zoom1.txt 2>&1
export NAX_M=256,222,1024,2048
for sh in qo kv gu down; do
  $W/target/release/examples/gemm_nax bench $sh > $R/bench1-$sh.txt 2>&1
done
echo done
