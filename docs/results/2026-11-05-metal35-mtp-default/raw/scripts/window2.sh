#!/bin/zsh
# window 2: prefill-focused ABAB (filler and natural-v1 text at 4k and 16k, 5 reps)
export OUT=ab-prefill SIZES=4096,16384 GEN=16 REPS=5 NATSIZES=4096,16384 NATREPS=5 NOIDS=1
zsh /opt/code/intarsiaml/.worktrees/flip-metal-mtp/.scratch/ab.sh
