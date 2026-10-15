#!/bin/zsh
# burst 6: in-engine tile check at chunk 512 / 384 / 128 (rule range upper end), 2 repeats, tiles in ABBA-ish order
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
R=$W/docs/results/raw/2026-10-15-metal-prefill-p45
export CFGS="4k:512 4k:384 4k:128"
$R/engine-sweep2.sh t128x64w4 t64x64w4 t64x64w2 t32x128w2 t32x32w1
