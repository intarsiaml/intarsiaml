#!/bin/zsh
D=/opt/code/intarsiaml/.claude/worktrees/agent-a492fe7bd7af6756c/docs/results/raw/2026-10-15-metal-decode-d1
B=/Users/josef/.local/state/claude-watch/d1bin
$D/abrun.sh short 4 ab2-short $B/intarsia-d1c base glu gluN all > $D/ab2-short.log 2>&1
$D/abrun.sh 4k 4 ab2-4k $B/intarsia-d1c base glu gluN all > $D/ab2-4k.log 2>&1
