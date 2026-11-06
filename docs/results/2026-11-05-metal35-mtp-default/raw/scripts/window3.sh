#!/bin/zsh
W=/opt/code/intarsiaml/.worktrees/flip-metal-mtp
echo "#### window2 $(date +%T)"; zsh $W/.scratch/window2.sh
echo "#### gates $(date +%T)"; zsh $W/.scratch/gates.sh
echo "#### done $(date +%T)"
