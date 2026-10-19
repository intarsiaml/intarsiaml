#!/usr/bin/env bash
# Mutation check of tests/gdn_split.rs on spark1 with vLLM UP (the mutants run the small correctness tests only):
#   fleetq run --host spark1 --mem 12 --label "prefill-4 mutation check" -- scripts/cuda-prefill4/mutcheck.sh [mutation ...]
# Syncs and builds the working tree as ~/intarsia-decfuse/p4-dev, then per mutation: restore the kernel source into p4-mut,
# apply mutate.py, build the fp8_gemm_reg test binary, run it. The unmutated tree is built and run first (must pass). Prints
# one KILLED / SURVIVED line each (mutations marked EQUIVALENT in mutate.py are expected to survive).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
ALL=$(python3 "$REPO/scripts/cuda-prefill4/mutate.py" --list)
MUTS=${*:-$ALL}
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
D='~/intarsia-decfuse'
"$REPO/scripts/cuda-decode/build.sh" p4-dev --tests > /dev/null 2>&1
rsh "mkdir -p $D/p4-mut && rsync -a --delete $D/p4-dev/ $D/p4-mut/" > /dev/null
scp -q "$REPO/scripts/cuda-prefill4/mutate.py" "$HOST:/tmp/p4_mutate_$FLEETQ_JOB.py"
bad=0
for m in unmutated $MUTS; do
  out=$(rsh "cd $D/p4-mut && cp $D/p4-dev/kernels/cuda/gdn.cu kernels/cuda/ && { [ $m = unmutated ] || python3 /tmp/p4_mutate_$FLEETQ_JOB.py . $m; } && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 timeout 900 cargo test --release -p intarsia-backend-cuda --test gdn_split -- --test-threads=1 2>&1 | grep -E '^(applied|mutation|test |error)|could not compile|panicked' | cut -c1-200")
  echo "--- $m"; echo "$out" | grep -E '^(applied|mutation)|^error(\[|: could not)' | head -3
  failed=$(echo "$out" | grep -E '^test .*FAILED' | sed 's/^test //; s/ \.\.\. FAILED//' | tr '\n' ' ')
  passed=$(echo "$out" | grep -cE '^test .* ok')
  if [ "$m" = unmutated ]; then
    if [ -z "$failed" ] && [ "$passed" -ge 2 ]; then echo "unmutated: pass ($passed tests)"; else echo "unmutated: FAILED: $failed"; bad=1; fi
  elif echo "$out" | grep -qE '^mutation|^error(\[|: could not)|could not compile'; then echo "HARNESS: $m did not apply or build"; bad=1
  elif [ -n "$failed" ]; then echo "KILLED $m: $failed"
  elif [ "$passed" -lt 2 ]; then echo "KILLED $m: only $passed tests completed (crash or hang)"
  else echo "SURVIVED $m: the tests passed on the mutated kernel"; echo "$out" | grep -q EQUIVALENT || bad=1; fi
done
rsh "rm -f /tmp/p4_mutate_$FLEETQ_JOB.py"
exit $bad
