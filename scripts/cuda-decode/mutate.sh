#!/usr/bin/env bash
# Mutation checks of the fused CUDA kernels, on spark1 with vLLM UP (correctness only), inside a shared fleetq slot:
#   fleetq run --host spark1 --mem 6 --label "..." -- scripts/cuda-decode/mutate.sh [mutation ...]
# For each mutation: copy the tree under test (~/intarsia-decfuse/dev) kernel sources over ~/intarsia-decfuse/mut, apply the
# mutation (mutate.py), build, run `--test fused_decode`. The test MUST fail (cargo test exit 101, "test result: FAILED"); a
# compile error or a pass is reported as a harness problem / a test that missed the bug. Finally the unmutated sources are
# restored and the test must pass again.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
ALL="pair-block pair-nb0 fp8-pair-scale fp8-pair-out glu-scale glu-round glu-blockscale gated-order gated-round gated-round-scalar"
MUTS=${*:-$ALL}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
D='~/intarsia-decfuse'
rsh "mkdir -p $D/mut && cd $D/mut && { test -d target || cp -a $D/dev/target target; }; rsync -a --delete --exclude target $D/dev/ $D/mut/" >/dev/null
T="cd $D/mut && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --test fused_decode 2>&1"
restore() { rsh "cd $D/mut && cp $D/dev/kernels/cuda/{matmul,fp8,nvfp4,norm}.cu kernels/cuda/"; }
bad=0
for m in $MUTS; do
  restore
  if ! rsh "cd $D/mut && python3 -I scripts/cuda-decode/mutate.py . $m"; then echo "HARNESS: $m did not apply"; bad=1; continue; fi
  out=$(rsh "$T | grep -E '^test |test result|error(\[|:)|panicked' | cut -c1-200")
  if echo "$out" | grep -q "error"; then echo "HARNESS: $m does not build"; echo "$out" | head -5; bad=1; continue; fi
  if echo "$out" | grep -q "test result: FAILED"; then
    echo "KILLED   $m: $(echo "$out" | grep -E '^test .*FAILED' | sed 's/^test //; s/ \.\.\. FAILED//' | tr '\n' ' ')"
  else
    echo "SURVIVED $m: the tests passed on a deliberately broken kernel"; bad=1
  fi
done
restore
out=$(rsh "$T | grep -E 'test result'")
echo "unmutated: $out"
echo "$out" | grep -q "test result: ok" || bad=1
exit $bad
