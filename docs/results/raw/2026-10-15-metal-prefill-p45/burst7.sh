#!/bin/zsh
# burst 7: qkv_fold tests (prefill qk_norm_rope_kv head counts), P4 triple-GEMM proxy (one 6144-column kernel) vs q + k/v pair
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
R=$W/docs/results/raw/2026-10-15-metal-prefill-p45
mkdir -p $R/qkv
cd $W
cargo test -p intarsia-backend-metal --release --test qkv_fold > $R/gates/test-qkv-fold.txt 2>&1; echo "qkv_fold exit $?"
export NAX_M=256,222,2048 NAX_POOL_MIB=3072 NAX_ONLY=t64x64w2,t32x128w2,t32x32w1,t128x64w4,t64x64w4
$W/target/release/examples/gemm_nax bench qkv > $R/qkv/bench-qkv.txt 2>&1
$W/target/release/examples/gemm_nax pick2 > $R/qkv/pick2-again.txt 2>&1
echo done
