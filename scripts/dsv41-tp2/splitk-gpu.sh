#!/usr/bin/env bash
# Lane dsv41/tp2: GPU checks of matmul_exl3_splitk on spark2 after build.sh (the tree is in $SPARK_DIR/src):
#   1. the existing EXL3 kernels' SASS is unchanged (old exl3.cu from git vs the new file, cuobjdump per kernel);
#   2. the exl3 test binary (split-K sweep vs the cpu reference, scratch mirror, and every existing EXL3 test);
#   3. the admission sweep of matmul_exl3 (all three kernels) and the DSv4.1 step-resolution test;
#   4. split-K vs GEMV timing on DSv4.1's small-grid decode shapes (bench_splitk, ignored test).
#   fleetq run --host spark2 --quiet gpu,cpu --mem 12 --expect 25 --label "dsv41 tp2 splitk GPU checks" -- scripts/dsv41-tp2/splitk-gpu.sh OLD_REV
source "$(dirname "${BASH_SOURCE[0]}")/spark-dir.sh"
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OLD=${1:-HEAD}
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; export MAX_JOBS=4 CARGO_BUILD_JOBS=4;'
rsh() { ssh -n -o BatchMode=yes spark2 "$RP export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
git -C "$REPO" show "$OLD:kernels/cuda/exl3.cu" | ssh -o BatchMode=yes spark2 'mkdir -p '"$SPARK_DIR"'/sass && cat > '"$SPARK_DIR"'/sass/exl3_old.cu'
echo "== 1. SASS of the existing kernels (old $OLD vs new)"
rsh 'cd '"$SPARK_DIR"'/sass && cp ../src/kernels/cuda/exl3.cu exl3_new.cu && for v in old new; do
  nvcc -cubin -arch=sm_121 -O3 -std=c++17 -I../src/kernels -I../src/kernels/cuda -o $v.cubin exl3_$v.cu 2>&1 | grep -v warning | head -5
  cuobjdump -sass $v.cubin | awk "/Function :/{f=\$3} f && !/Function :/{print f\"\\t\"\$0}" | grep -vE "split_kernel|split_finish" | sed -E "s/_GLOBAL__N__[0-9a-f]+_[0-9]+_exl3_(old|new)_cu_[0-9a-f]+/ANON/g; s/\/\*[0-9a-f]{4}\*\///" > $v.sass
done; n_old=$(cut -f1 old.sass | sort -u | wc -l); n_new=$(cut -f1 new.sass | grep -vE "split" | sort -u | wc -l)
echo "functions old $n_old new (without split-K) $n_new"; if cmp -s old.sass new.sass; then echo "SASS of every existing kernel: IDENTICAL"; else echo "SASS DIFFERS"; diff old.sass new.sass | head -20; fi'
echo "== 2. exl3 test binary"
rsh 'cd '"$SPARK_DIR"'/src && INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test exl3 2>&1 | grep -E "^test |worst|cases|test result|panicked|^error|error\[" | head -40'
echo "== 3. admission sweep + DSv4.1 step resolution"
rsh ''"$SPARK_DIR"'/bin/intarsia kernels test --backend cuda --op matmul_exl3 2>&1 | tail -8'
rsh 'cd '"$SPARK_DIR"'/src && cargo test --release -p intarsia-models --test dsv41 2>&1 | grep -E "test result|FAILED|panicked|^error|error\[" | head'
echo "== 4. split-K vs GEMV on DSv4.1 decode shapes"
rsh 'cd '"$SPARK_DIR"'/src && INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test exl3 -- --ignored --nocapture bench_splitk 2>&1 | grep -E "splitk|gemv|K=|test result|^error|error\[" | head -30'
