#!/bin/zsh
# plan show, kernels test, accuracy: main binary vs peel branch binary. From the worktree root, inside a fleetq slot.
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m27-flip
O=$PWD/.scratch/logs/gates; mkdir -p $O/plan $O/kernels $O/accuracy
B=$PWD/.scratch/intarsia.main; F=$PWD/.scratch/intarsia.peel
W=/Users/josef/models; M=$W/Qwen3.8-27B-Quark-AWQ-MXFP4
echo "== plan show"
for mdl in Qwen3.6-35B-A3B-NVFP4 Qwen3-8B Qwen3.8-27B-Quark-AWQ-MXFP4; do for fmt in "" "--json"; do
  $B plan show --model $W/$mdl --backend metal $fmt > $O/plan/$mdl.main$fmt.txt 2> $O/plan/$mdl.main$fmt.err
  $F plan show --model $W/$mdl --backend metal $fmt > $O/plan/$mdl.peel$fmt.txt 2> $O/plan/$mdl.peel$fmt.err
  if cmp -s $O/plan/$mdl.main$fmt.txt $O/plan/$mdl.peel$fmt.txt; then echo "plan show $mdl $fmt: identical"; else echo "plan show $mdl $fmt: DIFFERS"; diff $O/plan/$mdl.main$fmt.txt $O/plan/$mdl.peel$fmt.txt | cut -c1-200 | head -12; fi
done; done
echo "== kernels test"
$B kernels test --backend metal > $O/kernels/main.log 2>&1; echo "main kernels test exit $?"
$F kernels test --backend metal > $O/kernels/peel.log 2>&1; echo "peel kernels test exit $?"
tail -2 $O/kernels/peel.log | cut -c1-160
diff $O/kernels/main.log $O/kernels/peel.log | cut -c1-160 | head -10
echo "== accuracy 27B"
G=docs/results/2026-11-02-m27-metal/raw/golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
BL=golden/baselines/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.metal.baseline.json
for t in main peel; do
  [ $t = main ] && X=$B || X=$F
  $X accuracy --backend metal --model $M --max-seq 4096 --mxfp4-act none --golden $G --mode both --decode-positions 16 --baseline $BL --allow-unstamped > $O/accuracy/$t.out 2> $O/accuracy/$t.err; echo "accuracy $t exit $?"
  grep -E "verdict|golden gate|floor|baseline" $O/accuracy/$t.out | cut -c1-180 | tail -6
done
cmp -s $O/accuracy/main.out $O/accuracy/peel.out && echo "accuracy: main and peel byte-identical" || { echo "accuracy: main and peel DIFFER"; diff $O/accuracy/main.out $O/accuracy/peel.out | cut -c1-180 | head -30; }
