#!/bin/zsh
# W4A16 and W4A4, fused vs pre-pass (XFUSE 1 / 0), 3 shapes, M=1,3, 2 rounds. usage: bw6.sh OUT BIN
O=$1; B=$2; : > $O
for r in 1 2; do for act in 0 1; do for m in 1 3; do for xf in 1 0; do for sh in 17408x5120 5120x17408 1024x5120; do
  echo "== r$r act=$act M=$m XFUSE=$xf" >> $O; MXFP4_BW_ACT=$act MXFP4_BW_M=$m MXFP4_BW_SHAPE=$sh INTARSIA_METAL_MXFP4_XFUSE=$xf $B >> $O 2>&1
done; done; done; done; done
