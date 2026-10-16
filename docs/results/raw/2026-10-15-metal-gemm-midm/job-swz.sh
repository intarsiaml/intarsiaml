#!/bin/zsh
cd /opt/code/intarsiaml-gemm-midm
D=docs/results/raw/2026-10-15-metal-gemm-midm
for POOL in 3072 0; do
  NAX_M=256,512 NAX_SWZ=0,1,2 NAX_POOL_MIB=$POOL NAX_ONLY=t64x64w2,t32x128w2,t64x64w4 NAX_ROUNDS=21 $D/withclock.sh $D/swz-pool$POOL target/release/examples/gemm_nax bench qo down gu kv
done
