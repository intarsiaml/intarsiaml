#!/bin/bash
# G5 extra models: plan show base vs cand
N=/opt/code/intarsiaml/.worktrees/nxp/.nxp-out; mkdir -p $N/plans2
for m in Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct Qwen3.8-27B-Quark-AWQ-MXFP4 qwen38-27b; do
  for t in base cand; do $N/intarsia-$t plan show --model $HOME/models/$m --backend metal --max-seq 8192 > $N/plans2/$t-$m.txt 2>&1; echo "$t $m rc=$?"; done
  cmp -s $N/plans2/base-$m.txt $N/plans2/cand-$m.txt && echo "$m IDENTICAL" || echo "$m DIFFERS"
done
