#!/bin/bash
# fake "intarsia" binary for remote.sh: kernel bench A/B of the original and the fixed grouped MoE kernel
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsia-acc; cd $R/new
for round in 1 2; do
  for v in orig fixed; do
    cp $R/mg_$v.cu kernels/cuda/moe_grouped.cu
    echo "=== $v round $round"
    CARGO_TARGET_DIR=target/acc CARGO_BUILD_JOBS=8 cargo test --release -p intarsia-backend-cuda --test quantmoe bench_real_shapes -- --ignored --nocapture 2>&1 | grep -v "^warning\|^ *|\|^ *=\|^$\|-->" | tail -30
  done
done
cp $R/mg_fixed.cu kernels/cuda/moe_grouped.cu
