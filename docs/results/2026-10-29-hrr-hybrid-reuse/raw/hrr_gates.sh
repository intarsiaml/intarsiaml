#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/hrr
N=.scratch/target/release/intarsia; M=.scratch/target-main/release/intarsia
for m in Qwen3.6-35B-A3B-NVFP4 Qwen3-8B; do
  $N plan show --model ~/models/$m --backend metal > .scratch/plan-new-$m.txt 2>&1
  $M plan show --model ~/models/$m --backend metal > .scratch/plan-main-$m.txt 2>&1
  if diff -q .scratch/plan-new-$m.txt .scratch/plan-main-$m.txt >/dev/null; then echo "PLAN $m IDENTICAL ($(wc -l < .scratch/plan-new-$m.txt) lines)"; else echo "PLAN $m DIFFERS"; diff .scratch/plan-new-$m.txt .scratch/plan-main-$m.txt | head -10; fi
done
$N kernels test --backend metal > .scratch/kernels.txt 2>&1; echo "kernels exit $? FAIL count: $(grep -c FAIL .scratch/kernels.txt)"; tail -3 .scratch/kernels.txt
$N accuracy --model ~/models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode reuse --decode-positions 64 --max-seq 8192 --prefill-chunk 1024 --reuse-require-all > .scratch/acc-reuse-new.txt 2>&1; echo "accuracy exit $?"
