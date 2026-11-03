#!/bin/zsh
# W4A4: fused vs pre-pass, M=1,3, shapes 17408x5120 and 1024x5120 and 5120x17408; 2 rounds
O=$1; B=$2; : > $O
for r in 1 2; do for m in 1 3; do for xf in 1 0; do for sh in 17408x5120 5120x17408 1024x5120; do
  echo "== r$r M=$m XFUSE=$xf" >> $O; MXFP4_BW_ACT=1 MXFP4_BW_M=$m MXFP4_BW_SHAPE=$sh INTARSIA_METAL_MXFP4_XFUSE=$xf $B >> $O 2>&1
done; done; done; done
