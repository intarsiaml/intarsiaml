#!/bin/zsh
# usage: kt.sh OUTDIR BASEBIN NEWBIN ; kernels test --backend metal full, base and new (HSB unset) and new with HSB=3
O=$1; B=$2; N=$3; mkdir -p $O
$B kernels test --backend metal > $O/base-full.log 2>&1; echo "base exit $?"
$N kernels test --backend metal > $O/new-full.log 2>&1; echo "new exit $?"
INTARSIA_METAL_ATTN_HSB=3 $N kernels test --backend metal > $O/new-hsb3-full.log 2>&1; echo "new hsb3 exit $?"
INTARSIA_METAL_ATTN_HSB=3 $N kernels test --backend metal --op attn_decode -v > $O/new-hsb3-attn_decode-v.log 2>&1; echo "attn_decode -v exit $?"
