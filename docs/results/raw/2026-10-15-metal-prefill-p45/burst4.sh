#!/bin/zsh
# burst 4: t3 = t2 + 64x64w4 for 4096-column outputs at 512 < M <= 1536. tests, traces, plain ABBA pre vs t3
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p45
R=$W/docs/results/raw/2026-10-15-metal-prefill-p45
cd $W
cargo test -p intarsia-backend-metal --release --test gemm_nax --test qkv_fold > $R/test-t3.txt 2>&1
$R/trace.sh trace-t3 t3=$S/intarsia-t3
$R/plain-abba.sh plain-t3-1k 2048 1k 3 A=$S/intarsia-pre B=$S/intarsia-t3
$R/plain-abba.sh plain-t3-4k 2048,256 4k 2 A=$S/intarsia-pre B=$S/intarsia-t3
