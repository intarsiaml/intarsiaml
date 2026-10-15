#!/bin/zsh
# burst 1: pre (766c723) vs t2 (hybrid NAX tile pick + prefill qk_norm_rope_kv): kernel traces, plain prefill ABBA, qk lever isolation
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p45
R=$W/docs/results/raw/2026-10-15-metal-prefill-p45
$R/trace.sh trace-t2 pre=$S/intarsia-pre t2=$S/intarsia-t2
$R/plain-abba.sh plain-t2 256,2048 1k,4k,16k 2 A=$S/intarsia-pre B=$S/intarsia-t2
$R/plain-abba.sh plain-t2-qk 256 4k 2 A=$S/intarsia-t2@INTARSIA_METAL_NO_QK_PF=1 B=$S/intarsia-t2
