#!/bin/zsh
# test the NAX GEMM, then sweep tiles incl. the experimental ones
cd /opt/code/intarsiaml-gemm-midm
D=docs/results/raw/2026-10-15-metal-gemm-midm
cargo test --release -p intarsia-backend-metal --test gemm_nax -- --test-threads=1 > $D/test-gemm-nax-1.txt 2>&1
echo "test exit $?"
NAX_SHAPES=qo,kv,gu,down NAX_M=256,384,512,640,768,896,1024,1152,1280,1536,1792,2048 $D/withclock.sh $D/sweep3 target/release/examples/gemm_nax sweep
echo "sweep exit $?"
