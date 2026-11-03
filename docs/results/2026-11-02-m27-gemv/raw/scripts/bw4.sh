#!/bin/zsh
# usage: bw4.sh OUT BIN... : M=3 and M=8, 2 interleaved rounds, one shape set
O=$1; shift; : > $O
for r in 1 2; do for m in 3 8; do for b in "$@"; do echo "== r$r M=$m $(basename $b)" >> $O; MXFP4_BW_M=$m $b >> $O 2>&1; done; done; done
