#!/bin/zsh
# Speed cost of the peel with localeval, one quiet slot: main / peel / main / peel, single stream, sizes chosen so the prompt length is exposed (SIZES env), plus plain 4096 and 16384 controls.
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m27-flip
A=docs/results/2026-11-03-m27-hs-rowtable/raw/scripts/conc-arm.sh; O=.scratch/logs/speed; mkdir -p $O
export GEN=${GEN:-128} REPS=${REPS:-3}
n=0
for t in main peel main peel; do
  n=$((n+1))
  zsh $A $t-$n $PWD/.scratch/intarsia.$t $PWD/$O 1 1
done
