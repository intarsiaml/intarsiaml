#!/bin/zsh
# speed round 2: the sizes of the previous attempt (2099 / 2105 land in 2050..2057, 1063 / 1067 in 1026..1033, 6148 in 6146..6153) plus controls 2100 and 4096, in order peel, main, main, peel
# (round 1 ran main, peel, main, peel and its 4096 control drifted with arm order). Inside a fleetq slot (--quiet gpu,cpu).
set -u
cd /opt/code/intarsiaml/.worktrees/m35-peel-period
S=$PWD/.scratch; M35=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4; O=$S/logs/speed35b; mkdir -p $O
PICK="2099 2105 6148 1063 1067 2100 4096"
n=0
for t in peel main main peel; do n=$((n+1)); SIZES="$PICK" GEN=128 REPS=5 UNIM=1 zsh $S/speed-arm.sh $t-$n $S/intarsia.$t $O $M35; done
