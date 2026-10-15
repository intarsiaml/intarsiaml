#!/bin/zsh
# Experiment: P kept in float as the A operand of P.V (mixed float x bf16 matmul) instead of rounded to bf16.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
R=/opt/code/intarsiaml/.claude/worktrees/agent-ae91acc25118dab7a/docs/results/raw/2026-10-15-metal-prefill-f1/exp
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R
$S/flash_bench_pf32 --reps 6 nax 3840 2>&1 | tee $R/pf32-bench.txt
for B in pf32; do
  $S/intarsia-$B accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/$B-short.out 2> $R/$B-short.err
  $S/intarsia-$B accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $R/$B-long.out 2> $R/$B-long.err
  echo "$B short: $(grep -oE 'top-1 [0-9.]+%|full-vocab KL [0-9.e-]+|: (PASS|FAIL)' $R/$B-short.out | tr '\n' ' ')"
  echo "$B long : $(grep -oE 'top-1 [0-9.]+%|full-vocab KL [0-9.e-]+|: (PASS|FAIL)' $R/$B-long.out | tr '\n' ' ')"
done
