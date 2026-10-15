#!/bin/zsh
# Round 3: baseline-gated acc-4k. SAFE (256, matching baseline) and P3 scratch (2048 vs the 256 baseline: must say "not used", KL-only).
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k
W=/opt/code/intarsiaml-golden-4k
R=$W/docs/results/raw/2026-10-15-metal-golden-4k/round3
M=/Users/josef/models/Qwen3-8B
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 10 --label "metal golden-4k: acc-4k baseline-gated check" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
SK=(--skip kernels --skip gate --skip bench --skip bench-long --skip envelope --skip ab --skip tp --skip acc-short)
$S/intarsia-b qualify --model $M --backend metal --golden-dir $S/gold $SK --out $R/safe > $R/safe.log 2>&1; echo "safe exit $?"
$S/intarsia-p3b qualify --model $M --backend metal --golden-dir $S/gold $SK --out $R/p3-chunk-mismatch > $R/p3-chunk-mismatch.log 2>&1; echo "p3 exit $?"
echo done
