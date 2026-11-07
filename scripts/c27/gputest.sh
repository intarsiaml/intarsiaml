#!/usr/bin/env bash
# Lane c27, ON spark1 in one fleetq slot (vLLM up, small buffers, no timing claims): the CUDA backend tests that cover the pair kernel and the registry.
#   fleetq run --host spark1 --load cpu --mem 12 --expect 25 --label "c27 cuda gpu tests" -- ssh spark1 'TREE=dev2 bash ~/intarsia-c27/dev2/scripts/c27/gputest.sh'
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1
R=$HOME/intarsia-c27; T=${TREE:-dev2}; O=${O:-$R/out/gputest}; mkdir -p $O
cd $R/$T && export CARGO_TARGET_DIR=$R/target-$T INTARSIA_REQUIRE_GPU=1
for t in fused_decode rows_registry; do
  echo "== cargo test --release -p intarsia-backend-cuda --test $t $(date +%T)"
  cargo test --release -p intarsia-backend-cuda --test $t -- --test-threads=1 > $O/test-$t.log 2>&1; echo "exit $?"; grep -aE "^test |test result" $O/test-$t.log | tail -14
done
echo "== done $(date +%T)"
