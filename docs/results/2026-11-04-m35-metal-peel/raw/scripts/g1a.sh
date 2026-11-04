#!/bin/zsh
S=/opt/code/intarsiaml/.worktrees/m35-peel/.scratch; M35=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4
zsh $S/acc-run.sh $S/acc35/p $M35 "--max-seq 8192 --mode prefill" $S/acc35/oracle-p.safetensors main:$S/intarsia.main peel:$S/intarsia.peel peel0:$S/intarsia.peel:INTARSIA_PREFILL_PEEL=0
zsh $S/acc-run.sh $S/acc35/d $M35 "--max-seq 8192 --mode decode --decode-positions 32" $S/acc35/oracle-d.safetensors main:$S/intarsia.main peel:$S/intarsia.peel peel0:$S/intarsia.peel:INTARSIA_PREFILL_PEEL=0
