#!/bin/zsh
# plan show (35B, 27B, 8B) and kernels test: main binary vs branch binary. Inside a fleetq slot.
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m35-peel
O=$PWD/.scratch/logs/gates; mkdir -p $O/plan $O/kernels
B=$PWD/.scratch/intarsia.main; F=$PWD/.scratch/intarsia.peel
W=/Users/josef/models
echo "== plan show"
for mdl in Qwen3.6-35B-A3B-NVFP4 Qwen3-8B Qwen3.8-27B-Quark-AWQ-MXFP4 Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct; do for fmt in "" "--json"; do
  $B plan show --model $W/$mdl --backend metal $fmt > $O/plan/$mdl.main$fmt.txt 2> $O/plan/$mdl.main$fmt.err
  $F plan show --model $W/$mdl --backend metal $fmt > $O/plan/$mdl.peel$fmt.txt 2> $O/plan/$mdl.peel$fmt.err
  if cmp -s $O/plan/$mdl.main$fmt.txt $O/plan/$mdl.peel$fmt.txt; then echo "plan show $mdl $fmt: identical"; else echo "plan show $mdl $fmt: DIFFERS"; diff $O/plan/$mdl.main$fmt.txt $O/plan/$mdl.peel$fmt.txt | cut -c1-200 | head -12; fi
done; done
echo "== kernels test"
$B kernels test --backend metal > $O/kernels/main.log 2>&1; echo "main kernels test exit $?"
$F kernels test --backend metal > $O/kernels/peel.log 2>&1; echo "peel kernels test exit $?"
tail -2 $O/kernels/peel.log | cut -c1-160
grep -c FAIL $O/kernels/peel.log
diff $O/kernels/main.log $O/kernels/peel.log | cut -c1-160 | head -10
