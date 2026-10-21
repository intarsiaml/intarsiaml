#!/bin/zsh
# G3 final slot: GPU gates, then interleaved A/B x3 (base vs dev).
D=/opt/code/intarsiaml/.claude/worktrees/agent-a34836167f72ea687/docs/results/raw/2026-10-21-g3-metal
zsh $D/gates.sh $D/gates > $D/gates.console 2>&1
echo "gates exit $?"
zsh $D/ab.sh ab > $D/ab.console 2>&1
echo "ab exit $?"
echo FINAL_DONE $(date -Iseconds)
