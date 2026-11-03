#!/bin/zsh
# usage: bw2.sh OUT BASE_BIN NEW_BIN : M=1,3,8, two interleaved rounds
O=$1; A=$2; B=$3; : > $O
for r in 1 2; do for m in 1 3 8; do for arm in a b; do
  if [ $arm = a ]; then BIN=$A; else BIN=$B; fi
  echo "== r$r M=$m $arm" >> $O; MXFP4_BW_M=$m $BIN >> $O 2>&1
done; done; done
