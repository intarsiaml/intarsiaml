#!/bin/zsh
# env overrides on the 35B (metal): period 256 (first attempt's cut), period 8 (refused), peel 0 (off), peel 16 with the recipe's period
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m35-peel-period
M=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4; B=$PWD/.scratch/intarsia.peel; O=$PWD/.scratch/logs/envcheck; mkdir -p $O
for spec in "INTARSIA_PREFILL_PEEL_PERIOD=256" "INTARSIA_PREFILL_PEEL_PERIOD=8" "INTARSIA_PREFILL_PEEL_PERIOD=0" "INTARSIA_PREFILL_PEEL=0" "INTARSIA_PREFILL_PEEL=16"; do
  echo "== $spec"; env $spec $B plan show --model $M --backend metal > $O/$spec.txt 2> $O/$spec.err; echo "exit $?"; grep -n "peel" $O/$spec.txt; tail -1 $O/$spec.err | cut -c1-250
done
