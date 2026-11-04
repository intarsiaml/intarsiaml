#!/bin/zsh
# CPU oracle (f32 kernels, the fx1 oracle config) of the 35B at the accuracy prompt lengths. Inside a fleetq slot (--load cpu).
cd /opt/code/intarsiaml/.worktrees/m35-peel
B=$PWD/.scratch/intarsia.main; M=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4; A=$PWD/.scratch/acc35
for s in d2; do
  X=(--mode prefill)
  date; $B accuracy --model $M --backend cpu --golden $A/ids-$s.safetensors $X --max-seq 8192 --mem-budget 32 --prefill-chunk 256 --allow-unstamped --allow-dirty --write-self-golden $A/oracle-$s.safetensors > $A/oracle-$s.out 2> $A/oracle-$s.err
  echo "oracle $s exit $?"; ls -la $A/oracle-$s.safetensors
done
date
