#!/bin/zsh
# usage: kt2.sh OUTDIR BASEBIN NEWBIN ; kernels test with the 27B config's shapes (24/4/256 in the stock attn sweeps), attn_decode and attn_prefill, base / new / new+HSB=3
O=$1; B=$2; N=$3; mkdir -p $O
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4
for op in attn_decode attn_prefill; do
  $B kernels test --backend metal -m $M --op $op -v > $O/m27-$op-base.log 2>&1; echo "base $op exit $?"
  $N kernels test --backend metal -m $M --op $op -v > $O/m27-$op-new.log 2>&1; echo "new $op exit $?"
  INTARSIA_METAL_ATTN_HSB=3 $N kernels test --backend metal -m $M --op $op -v > $O/m27-$op-new-hsb3.log 2>&1; echo "new hsb3 $op exit $?"
done
