#!/bin/zsh
# SwiGLU-epilogue prototype: correctness test (bitwise pair + silu_glu) and microbench (glu vs pair per tile and M)
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
R=$W/docs/results/raw/2026-10-15-metal-prefill-p45/glu
mkdir -p $R
cd $W
cargo test -p intarsia-backend-metal --release --test gemm_nax nax_glu > $R/test-glu.txt 2>&1
NAX_M=256,222,2048 $W/target/release/examples/gemm_nax glu > $R/bench-glu.txt 2>&1
echo done
