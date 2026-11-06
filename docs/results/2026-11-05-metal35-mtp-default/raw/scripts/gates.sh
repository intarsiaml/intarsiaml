#!/bin/zsh
# no-op checks for the other models: plan show (text + json) main vs branch for 5 models, kernels test, 256 greedy ids (short + ~3.5k prompt) for the 8B and the 27B, main vs branch
[ -n "${FLEETQ_JOB:-}" ] || exit 2
W=/opt/code/intarsiaml/.worktrees/flip-metal-mtp; cd $W
O=$W/.scratch/logs/gates; mkdir -p $O/plan $O/kernels $O/noop
B=$W/.scratch/intarsia.main; F=$W/.scratch/intarsia.branch; MD=/Users/josef/models
echo "== plan show"
for mdl in Qwen3.6-35B-A3B-NVFP4 Qwen3-8B Qwen3.8-27B-Quark-AWQ-MXFP4 Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct; do for fmt in "" "--json"; do
  $B plan show --model $MD/$mdl --backend metal $fmt > $O/plan/$mdl.main$fmt.txt 2> $O/plan/$mdl.main$fmt.err
  $F plan show --model $MD/$mdl --backend metal $fmt > $O/plan/$mdl.branch$fmt.txt 2> $O/plan/$mdl.branch$fmt.err
  if cmp -s $O/plan/$mdl.main$fmt.txt $O/plan/$mdl.branch$fmt.txt; then echo "plan show $mdl $fmt: identical"; else echo "plan show $mdl $fmt: DIFFERS"; diff $O/plan/$mdl.main$fmt.txt $O/plan/$mdl.branch$fmt.txt | cut -c1-200 | head -8; fi
done; done
echo "== kernels test"
$B kernels test --backend metal > $O/kernels/main.log 2>&1; echo "main kernels test exit $?"
$F kernels test --backend metal > $O/kernels/branch.log 2>&1; echo "branch kernels test exit $?"
tail -2 $O/kernels/branch.log | cut -c1-160; echo "FAIL lines: main $(grep -c FAIL $O/kernels/main.log) branch $(grep -c FAIL $O/kernels/branch.log)"
echo "== noop ids (8B, 27B)"
for t in main branch; do
  if [ $t = main ]; then R=$B; else R=$F; fi
  RUN=$R bash scripts/tp2-hip/noop/noop.sh run $O/noop $t metal $MD/Qwen3-8B $MD/Qwen3.8-27B-Quark-AWQ-MXFP4 scripts/tp2-hip/noop/prompts
done
bash scripts/tp2-hip/noop/noop.sh cmp $O/noop main branch
