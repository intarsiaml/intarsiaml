#!/bin/zsh
# burst 2: pre (766c723) vs t2 (hybrid NAX tile pick + prefill qk_norm_rope_kv), plain prefill A B B A, 2 rounds
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p45
R=$W/docs/results/raw/2026-10-15-metal-prefill-p45
$R/plain-abba.sh plain-t2 256,2048 1k,4k,16k 2 A=$S/intarsia-pre B=$S/intarsia-t2
