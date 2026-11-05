#!/bin/zsh
S=/opt/code/intarsiaml/.worktrees/m35-peel/.scratch; M27=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4
zsh $S/acc-run.sh $S/acc27/set2 $M27 "--max-seq 4096 --mxfp4-act none --mode both --decode-positions 64" $S/acc27/ref2-f32.safetensors pre297:$S/intarsia.pre297 main:$S/intarsia.main
zsh $S/others.sh
