#!/bin/zsh
# Round 2 (after acc-4k got --min-top1 98.0): fresh full qualify on SAFE and P3 with the 4k golden, and the acc stages only
# without a 4k golden (the unchanged path, and P3's original failure).
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k
W=/opt/code/intarsiaml-golden-4k
R=$W/docs/results/raw/2026-10-15-metal-golden-4k/round2
A=$S/intarsia-a2; P=$S/intarsia-p32
M=/Users/josef/models/Qwen3-8B
mkdir -p $R
export INTARSIA_CPU_FREE_MEM_GB=60
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 30 --label "metal golden-4k: qualify round 2 (SAFE+tailfix vs P3)" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
SK=(--skip kernels --skip gate --skip bench --skip bench-long --skip envelope --skip ab --skip tp)
echo "== 1. no 4k golden, acc stages only: SAFE (256), P3 (2048)"
$A qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $S/gold-no4k $SK --out $R/safe-no4k > $R/safe-no4k.log 2>&1; echo "safe-no4k exit $?"
$P qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $S/gold-no4k $SK --out $R/p3-no4k > $R/p3-no4k.log 2>&1; echo "p3-no4k exit $?"
echo "== 2. full qualify with the 4k golden"
$A qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $S/gold --out $R/safe-qualify > $R/safe-qualify.log 2>&1; echo "safe qualify exit $?"
$P qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $S/gold --out $R/p3-qualify > $R/p3-qualify.log 2>&1; echo "p3 qualify exit $?"
echo done
