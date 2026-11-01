#!/usr/bin/env bash
# Lane dsv41/tp2: GPU checks of the f16-weight dense kernels (kernels/cuda/dsv41_dense.cu) on spark2 after build.sh:
# old kernel (OLD_REV) microbench, then the new kernel's tests (cpu reference, row bits) and microbench.
#   fleetq run --host spark2 --quiet gpu,cpu --mem 14 --expect 20 --label "dsv41 tp2 f16w GPU checks" -- scripts/dsv41-tp2/dense-gpu.sh OLD_REV
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OLD=${1:-HEAD}
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; export MAX_JOBS=4 CARGO_BUILD_JOBS=4 INTARSIA_REQUIRE_GPU=1;'
rsh() { ssh -n -o BatchMode=yes spark2 "$RP export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsia-dsv41-graphs/src && $*"; }
T='cargo test --release -p intarsia-backend-cuda --test dsv41_dense'
git -C "$REPO" show "$OLD:kernels/cuda/dsv41_dense.cu" | ssh -o BatchMode=yes spark2 'cat > ~/intarsia-dsv41-graphs/dsv41_dense_old.cu'
echo "== old kernel ($OLD) microbench"
rsh "cp kernels/cuda/dsv41_dense.cu ../dsv41_dense_new.cu && cp ../dsv41_dense_old.cu kernels/cuda/dsv41_dense.cu && $T -- --ignored --nocapture bench 2>&1 | grep -E 'N=|error|panicked' ; cp ../dsv41_dense_new.cu kernels/cuda/dsv41_dense.cu"
echo "== new kernel: tests"
rsh "$T 2>&1 | grep -E '^ok|^FAIL|^test |test result|panicked|^error' | head -60"
echo "== new kernel microbench"
rsh "$T -- --ignored --nocapture bench 2>&1 | grep -E 'N=|error|panicked'"
echo "== admission sweep logits/matmul"
rsh 'for o in matmul logits; do ~/intarsia-dsv41-graphs/bin/intarsia kernels test --backend cuda --op $o --kernel ${o}_f16w 2>&1 | grep -E "f16w|kernels,"; done'
