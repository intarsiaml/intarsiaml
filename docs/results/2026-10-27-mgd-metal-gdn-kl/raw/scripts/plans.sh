#!/bin/bash
R=/opt/code/intarsiaml/.worktrees/mgd/run; O=$R/plans
for m in Qwen3-8B Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct Qwen3.8-27B-Quark-AWQ-MXFP4 qwen38-27b Qwen3.6-35B-A3B-NVFP4; do
  for t in base cand; do $R/intarsia-$t plan show --model $HOME/models/$m --backend metal --max-seq 8192 > $O/$t-$m.txt 2>&1; echo "$t $m rc=$?"; done
  cmp -s $O/base-$m.txt $O/cand-$m.txt && echo "$m IDENTICAL" || echo "$m DIFFERS"
done
