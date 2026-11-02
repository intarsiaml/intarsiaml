#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/hrr
N=.scratch/target/release/intarsia; M=.scratch/target-main/release/intarsia
for m in Qwen3.6-35B-A3B-NVFP4 Qwen3-8B; do
  $N plan show --model ~/models/$m --backend metal > .scratch/plan-new2-$m.txt 2>&1
  if diff -q .scratch/plan-new2-$m.txt .scratch/plan-main-$m.txt >/dev/null; then echo "PLAN $m IDENTICAL"; else echo "PLAN $m DIFFERS"; diff .scratch/plan-new2-$m.txt .scratch/plan-main-$m.txt | head; fi
done
$N kernels test --backend metal > .scratch/kernels2.txt 2>&1; echo "kernels exit $? FAIL count: $(grep -c FAIL .scratch/kernels2.txt)"
ACC="accuracy --model $HOME/models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode reuse --decode-positions 64 --max-seq 8192 --prefill-chunk 1024 --reuse-require-all"
$N $ACC > .scratch/acc-default.txt 2>&1; echo "accuracy default exit $?"
INTARSIA_PREFIX_REUSE=exact $N $ACC > .scratch/acc-exact.txt 2>&1; echo "accuracy exact exit $?"
export PROBE_ARGS=ab
.scratch/hrr_run.sh default 18611 INTARSIA_X=1
.scratch/hrr_run.sh exact 18612 INTARSIA_PREFIX_REUSE=exact
