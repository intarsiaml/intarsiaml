#!/bin/zsh
# scout prompt lengths, pick sizes, then main/peel/main/peel arms of localeval speed (single stream). Inside a fleetq slot (--quiet gpu,cpu).
set -u
cd /opt/code/intarsiaml/.worktrees/m35-peel
S=$PWD/.scratch; M35=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4; O=$S/logs/speed35; mkdir -p $O
SC=$(python3 -c "print(','.join([str(x) for x in list(range(2035,2076))+list(range(6130,6172,2))+list(range(1015,1047,2))]))")
SIZES="$SC" GEN=1 REPS=2 UNIM=0 zsh $S/speed-arm.sh scout $S/intarsia.main $O $M35
PICK=$(python3 $S/pick-sizes.py $O); echo "picked sizes: $PICK"
[ -n "$PICK" ] || exit 3
n=0
for t in main peel main peel; do n=$((n+1)); SIZES="$PICK" GEN=128 REPS=5 UNIM=1 zsh $S/speed-arm.sh $t-$n $S/intarsia.$t $O $M35; done
