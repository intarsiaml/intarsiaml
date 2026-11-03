#!/bin/zsh
# usage: bw.sh BIN OUT [variants...]  -- microbench base kernel and each INTARSIA_METAL_MXFP4_DEV variant
BIN=$1; O=$2; shift 2
echo "== base" > $O; $BIN >> $O 2>&1
for v in "$@"; do echo "== $v" >> $O; INTARSIA_METAL_MXFP4_DEV=$v $BIN >> $O 2>&1; done
