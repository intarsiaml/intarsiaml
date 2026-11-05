#!/bin/zsh
# prompt length of localeval's prompt per size (35B tokenizer): reps 2, gen 1, no unanimis log. Inside a fleetq slot.
cd /opt/code/intarsiaml/.worktrees/m35-peel
SIZES="${SCOUT_SIZES}" GEN=1 REPS=2 UNIM=0 zsh .scratch/speed-arm.sh scout $PWD/.scratch/intarsia.main $PWD/.scratch/logs/speed-scout /Users/josef/models/Qwen3.6-35B-A3B-NVFP4
