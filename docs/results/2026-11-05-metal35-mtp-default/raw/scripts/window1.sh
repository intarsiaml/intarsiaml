#!/bin/zsh
# window 1: greedy ids gate; only if it passes, the ABAB speed window
W=/opt/code/intarsiaml/.worktrees/flip-metal-mtp
echo "#### ids $(date +%T)"; zsh $W/.scratch/ids.sh 2>&1 | tee $W/.scratch/logs/ids.summary
if grep -q "^ids fails=0" $W/.scratch/logs/ids.summary; then echo "#### ab $(date +%T)"; zsh $W/.scratch/ab.sh; else echo "#### IDS DIFFER: ab not run"; fi
