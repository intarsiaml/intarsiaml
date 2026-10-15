#!/bin/zsh
# burst 5 (final binary = intarsia-t2, merged tree): per-shape microbench before/after, final plain-prefill ABBA, localeval ABBA
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p45
R=$W/docs/results/raw/2026-10-15-metal-prefill-p45
mkdir -p $R/micro
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
echo "idle: $(idle3)" | tee $R/micro/idle.txt
$W/target/release/examples/gemm_nax pick2 > $R/micro/pick2.txt 2>&1
$R/plain-abba.sh plain-final 256,2048 1k,4k 3 A=$S/intarsia-pre B=$S/intarsia-t2
$R/plain-abba.sh plain-final-16k 256,2048 16k 2 A=$S/intarsia-pre B=$S/intarsia-t2
$R/localeval-abba.sh localeval-c256 256
$R/localeval-abba.sh localeval-c2048 2048
