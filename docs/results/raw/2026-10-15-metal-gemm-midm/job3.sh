#!/bin/zsh
cd /opt/code/intarsiaml-gemm-midm
D=docs/results/raw/2026-10-15-metal-gemm-midm
NAX_ONLY=t128x64w4,t64x64w2,t32x128w2 NAX_ROUNDS=25 NAX_SHAPES=qo,kv,gu,down NAX_M=640,768,896,1024,1152,1280,1408,1536,1664,1792,1920,2048 $D/withclock.sh $D/sweep4 target/release/examples/gemm_nax sweep
echo "sweep exit $?"
