#!/bin/zsh
# gates 2: kernels test --backend metal, final and base. usage: gates2.sh OUTROOT FINALBIN BASEBIN
O=$1; F=$2; B=$3; mkdir -p $O/kernels
$F kernels test --backend metal > $O/kernels/final-full.log 2>&1; echo "final kernels test exit $?"
$B kernels test --backend metal > $O/kernels/base-full.log 2>&1; echo "base kernels test exit $?"
