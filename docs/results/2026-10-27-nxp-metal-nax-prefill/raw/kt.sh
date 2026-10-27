#!/bin/bash
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; G=/opt/code/intarsiaml/golden; N=/opt/code/intarsiaml/.worktrees/nxp/.nxp-out; C=$N/intarsia-cand; B=$N/intarsia-base; mkdir -p $N/kt
$C kernels test --backend metal --model $M > $N/kt/cand-35b.out 2>&1; echo "kernels test (35B shapes) exit $?"; tail -3 $N/kt/cand-35b.out
$C kernels test --backend metal > $N/kt/cand-default.out 2>&1; echo "kernels test (default shapes) exit $?"; tail -3 $N/kt/cand-default.out
for a in base=$B cand=$C; do tag=${a%%=*}; bin=${a#*=}
  $bin accuracy --model $M --backend metal --golden $G/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode both --decode-positions 256 --max-seq 8192 --write-verdict $N/kt/$tag-4k256.verdict.json > $N/kt/$tag-4k256.log 2>&1; echo "$tag 4k decode-256 exit $?"
done
