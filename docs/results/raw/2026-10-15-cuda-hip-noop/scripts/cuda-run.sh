#!/bin/bash
# usage: cuda-run.sh <A|B|BASE>   (runs on spark1 inside fleetq job 2002; CPU build + small GPU tests)
T=$1
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_PATH=/usr/local/cuda CARGO_BUILD_JOBS=5
export CARGO_TARGET_DIR=/tmp/cnc/$T-target
OUT=/tmp/cnc/out-$T; mkdir -p $OUT
cd /tmp/cnc/$T || exit 9
{ echo "head: $(cat COMMIT 2>/dev/null)"; rustc --version; nvcc --version | tail -2; } > $OUT/versions.txt 2>&1
cargo build --release -p intarsia-cli --features cuda > $OUT/build.log 2>&1; echo "EXIT $?" >> $OUT/build.log
for c in intarsia-core intarsia-models; do
  cargo test --release -p $c > $OUT/test-$c.log 2>&1; echo "EXIT $?" >> $OUT/test-$c.log
done
INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda > $OUT/test-intarsia-backend-cuda.log 2>&1; echo "EXIT $?" >> $OUT/test-intarsia-backend-cuda.log
cargo test --release -p intarsia-cli --features cuda > $OUT/test-intarsia-cli.log 2>&1; echo "EXIT $?" >> $OUT/test-intarsia-cli.log
echo ALLDONE > $OUT/DONE
