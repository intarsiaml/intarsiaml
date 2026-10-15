#!/bin/zsh
# Re-run of the qualify stages that final_gates.sh could not run (acc-short, acc-long, envelope were SKIP: the worktree
# has no golden/ directory). Resumes the earlier output directory; run inside a fleetq slot.
W=/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7
R=$W/docs/results/raw/2026-10-14-metal-prefill-p1/full/final
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
cd $W
./target/release/intarsia qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir /opt/code/intarsiaml/golden --out $R/qualify-golden > $R/qualify-golden-console.log 2>&1; echo "qualify exit $?"
echo done
