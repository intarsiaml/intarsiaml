#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/mgd
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
mkdir -p run/mutout
for m in "$@"; do
  ./run/mut/intarsia-$m kernels test --backend metal --model $M --op gdn_delta_rule --kernel gdn_delta_rule_bf16_pre > run/mutout/$m.out 2>&1
  echo "$m exit=$? $(grep 'gdn_delta_rule_bf16_pre' run/mutout/$m.out | tail -1)"
done
