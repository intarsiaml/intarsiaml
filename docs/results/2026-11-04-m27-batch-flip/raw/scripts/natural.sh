#!/bin/zsh
# unmodified ident-rt.sh burst at max-batch 2, with the existing scheduler trace on
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m27-flip
export INTARSIA_CB_TRACE=1 ROUNDS=${ROUNDS:-6}
zsh docs/results/2026-11-03-m27-hs-rowtable/raw/scripts/ident-rt.sh .scratch/logs/natural $1 $PWD/.scratch/intarsia.head 2
