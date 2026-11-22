#!/bin/bash
# quiet: rerun of diagx with the built binaries: A B A B B A B A, R 8/16, box check at start and end.
trap 'kill 0' TERM
O=/opt/code/intarsiaml/.worktrees/ink-batch-34/.scratch; W=/opt/code/intarsiaml/.worktrees/ink-batch
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
[ -x $W/.scratch/alucheck/alucheck ] && $W/.scratch/alucheck/alucheck 3 | tail -2
$O/mb_$1 --reps 60 --rows 16 proj > /dev/null
for t in $1 $2 $1 $2 $2 $1 $2 $1; do echo "== tree $t"; $O/mb_$t --reps 80 --rows 8,16 proj | grep 'S=1'; done
[ -x $W/.scratch/alucheck/alucheck ] && $W/.scratch/alucheck/alucheck 3 | tail -2
echo "diagx2 done"
