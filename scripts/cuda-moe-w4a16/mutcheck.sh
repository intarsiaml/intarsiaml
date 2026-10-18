#!/usr/bin/env bash
# Mutation check of tests/moe_w4a16.rs on spark1 with vLLM UP (the mutants run the small correctness tests only):
#   fleetq run --host spark1 --mem 12 --label "w4a16 mutation check" -- scripts/cuda-moe-w4a16/mutcheck.sh [mutation ...]
# Syncs and builds the working tree as ~/intarsia-decfuse/w4-dev, then per mutation: restore the kernel source into w4-mut,
# apply mutate.py, build the moe_w4a16 test binary, run it. The unmutated tree is built and run first (must pass). Prints
# one KILLED / SURVIVED line each.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
ALL=$(python3 "$REPO/scripts/cuda-moe-w4a16/mutate.py" --list)
MUTS=${*:-$ALL}
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
D='~/intarsia-decfuse'
"$REPO/scripts/cuda-decode/build.sh" w4-dev --tests > /dev/null 2>&1
rsh "mkdir -p $D/w4-mut && rsync -a --delete $D/w4-dev/ $D/w4-mut/" > /dev/null
scp -q "$REPO/scripts/cuda-moe-w4a16/mutate.py" "$HOST:/tmp/w4a16_mutate_$FLEETQ_JOB.py"
bad=0
for m in unmutated $MUTS; do
  out=$(rsh "cd $D/w4-mut && cp $D/w4-dev/kernels/cuda/moe_w4a16.cu kernels/cuda/ && { [ $m = unmutated ] || python3 /tmp/w4a16_mutate_$FLEETQ_JOB.py . $m; } && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 timeout 900 cargo test --release -p intarsia-backend-cuda --test moe_w4a16 -- --test-threads=1 2>&1 | grep -E '^(applied|mutation|test |error)|could not compile' | cut -c1-160")
  echo "--- $m"; echo "$out" | grep -E '^(applied|mutation)|^error(\[|: could not)' | head -3
  failed=$(echo "$out" | grep -E '^test .*FAILED' | sed 's/^test //; s/ \.\.\. FAILED//' | tr '\n' ' ')
  passed=$(echo "$out" | grep -cE '^test .* ok')
  if [ "$m" = unmutated ]; then
    if [ -z "$failed" ] && [ "$passed" -ge 4 ]; then echo "unmutated: pass ($passed tests)"; else echo "unmutated: FAILED: $failed"; bad=1; fi
  elif echo "$out" | grep -qE '^mutation|^error(\[|: could not)|could not compile'; then echo "HARNESS: $m did not apply or build"; bad=1
  elif [ -n "$failed" ]; then echo "KILLED $m: $failed"
  elif [ "$passed" -lt 4 ]; then echo "KILLED $m: only $passed tests completed (crash or hang)"
  else echo "SURVIVED $m: the tests passed on a deliberately broken kernel"; bad=1; fi
done
rsh "rm -f /tmp/w4a16_mutate_$FLEETQ_JOB.py"
exit $bad
