#!/bin/zsh
D=/opt/code/intarsiaml/.claude/worktrees/agent-a492fe7bd7af6756c/docs/results/raw/2026-10-15-metal-decode-d1
B=/Users/josef/.local/state/claude-watch/d1bin
$B/glu-test-3 --test-threads=1 > $D/tests/glu-d1d.log 2>&1
echo "glu tests exit $?"
$D/abrun.sh short 5 ab3-short $B/intarsia-d1d base glu glu256 > $D/ab3-short.log 2>&1
$D/abrun.sh 4k 4 ab3-4k $B/intarsia-d1d base glu glu256 > $D/ab3-4k.log 2>&1
