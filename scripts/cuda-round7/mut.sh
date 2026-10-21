#!/usr/bin/env bash
# Round-7 checks on spark1 (vLLM up, small slot), inside the caller's fleetq slot:
#   fleetq run --host spark1 --mem 8 --label "..." -- scripts/cuda-round7/mut.sh
# 1. syncs + builds this tree as ~/intarsia-decfuse/r7-mbase (seeded from r7-dev's target dir);
# 2. racecheck of every race-targets.txt line (the new lt line included) on r7-mbase;
# 3. `intarsia kernels test --backend cuda` (admission sweep) with INTARSIA_CUDA_FP8_GEMM_LT=1 and without;
# 4. scripts/cuda-prefill4/mutcheck.sh with scripts/cuda-round7/mutate.py: launcher mutants (tests/quantmoe.rs, racecheck
#    on the lt line), gate mutant (kernels.rs, --lib), range mutant (quantmoe.rs, --lib).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
STEPS=${STEPS:-race,adm,mut}
has() { [[ ",$STEPS," == *",$1,"* ]]; }
rsh() { ssh -n -o BatchMode=yes spark1 "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "D=~/intarsia-decfuse; mkdir -p \$D/r7-mbase; test -d \$D/r7-mbase/target || cp -a \$D/r7-dev/target \$D/r7-mbase/target"
"$REPO/scripts/cuda-decode/build.sh" r7-mbase --tests 2>&1 | grep -E "^error|Finished|stamp" | head
if has race; then
  echo "===== racecheck (all targets)"
  "$REPO/scripts/cuda-checks/racecheck.sh" r7-mbase; echo "racecheck exit $?"
fi
if has adm; then
  for e in INTARSIA_X=0 INTARSIA_CUDA_FP8_GEMM_LT=1; do
    echo "===== kernels test --backend cuda ($e)"
    rsh "cd ~/intarsia-decfuse/r7-mbase && env $e timeout 900 target/acc/release/intarsia kernels test --backend cuda 2>&1 | grep -E 'matmul_fp8|FAIL|REJECT|kernels,|cases|summary|total' | tail -15"
  done
fi
if has mut; then
  P=$S/mutate.py
  echo "===== launcher mutants: $(python3 -I "$P" --list fp8_gemm_lt.cu)"
  MUT_PY=$P KFILE=kernels/cuda/fp8_gemm_lt.cu TESTS="--test quantmoe" MIN_PASS=10 BASE_TREE=r7-mbase MUT_TREE=r7-mut NOBUILD=1 \
    "$REPO/scripts/cuda-prefill4/mutcheck.sh" $(python3 -I "$P" --list fp8_gemm_lt.cu)
  echo "===== gate mutant"
  MUT_PY=$P KFILE=crates/intarsia-backend-cuda/src/kernels.rs TESTS="--lib" MIN_PASS=13 BASE_TREE=r7-mbase MUT_TREE=r7-mut RACECHECK=0 NOBUILD=1 \
    "$REPO/scripts/cuda-prefill4/mutcheck.sh" $(python3 -I "$P" --list kernels.rs)
  echo "===== range mutant"
  MUT_PY=$P KFILE=crates/intarsia-backend-cuda/src/quantmoe.rs TESTS="--lib" MIN_PASS=13 BASE_TREE=r7-mbase MUT_TREE=r7-mut RACECHECK=0 NOBUILD=1 \
    "$REPO/scripts/cuda-prefill4/mutcheck.sh" $(python3 -I "$P" --list quantmoe.rs)
fi
