#!/usr/bin/env bash
# Mutation check of a CUDA kernel's tests on spark1 with vLLM UP (the mutants run the small correctness tests only):
#   fleetq run --host spark1 --mem 12 --label "prefill-4 mutation check" -- scripts/cuda-prefill4/mutcheck.sh [mutation ...]
# Syncs and builds the working tree as ~/intarsia-decfuse/$BASE_TREE, then per mutation: restore the kernel source into
# $MUT_TREE, apply $MUT_PY, build and run the tests. A mutant that passes the tests then goes through the shared-memory race
# check (scripts/cuda-checks/racecheck.sh on the race-targets.txt lines of $KFILE), the same step scripts/cuda-decode/gpu-test.sh
# runs: a dropped __syncthreads / __syncwarp the bitwise tests cannot see is KILLED there. The unmutated tree is built and run
# first (tests must pass, racecheck must be clean). Prints one KILLED (tests | racecheck) / SURVIVED line each (mutations marked
# EQUIVALENT in the mutate script are expected to survive).
# Env (defaults: the round-4 GDN split kernel): MUT_PY (scripts/cuda-prefill4/mutate.py), KFILE (kernels/cuda/gdn.cu),
#      TESTS ("--test gdn_split"), MIN_PASS (2), BASE_TREE (p4-dev), MUT_TREE (p4-mut), RACECHECK (1; 0 = tests only),
#      NOBUILD (0; 1 = $BASE_TREE is already synced and built).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MUT_PY=${MUT_PY:-$REPO/scripts/cuda-prefill4/mutate.py}
KFILE=${KFILE:-kernels/cuda/gdn.cu}
TESTS=${TESTS:---test gdn_split}
MIN_PASS=${MIN_PASS:-2}
BT=${BASE_TREE:-p4-dev}; MT=${MUT_TREE:-p4-mut}
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
ALL=$(python3 "$MUT_PY" --list)
MUTS=${*:-$ALL}
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
D='~/intarsia-decfuse'
race() { [[ "${RACECHECK:-1}" != 0 ]] || return 0; KFILE=$KFILE RACE_LINES=4 HOST=$HOST "$REPO/scripts/cuda-checks/racecheck.sh" "$MT"; }
[[ "${NOBUILD:-0}" == 1 ]] || "$REPO/scripts/cuda-decode/build.sh" "$BT" --tests > /dev/null 2>&1
rsh "mkdir -p $D/$MT && rsync -a --delete $D/$BT/ $D/$MT/" > /dev/null
MP=/tmp/mutate_${FLEETQ_JOB}_$(basename "$MUT_PY")
scp -q "$MUT_PY" "$HOST:$MP"
bad=0
for m in unmutated $MUTS; do
  out=$(rsh "cd $D/$MT && cp $D/$BT/$KFILE $KFILE && { [ $m = unmutated ] || python3 $MP . $m; } && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 timeout 900 cargo test --release -p intarsia-backend-cuda $TESTS -- --test-threads=1 2>&1 | grep -E '^(applied|mutation|test |error)|could not compile|panicked' | cut -c1-200")
  echo "--- $m"; echo "$out" | grep -E '^(applied|mutation)|^error(\[|: could not)' | head -3
  failed=$(echo "$out" | grep -E '^test .*FAILED' | sed 's/^test //; s/ \.\.\. FAILED//' | tr '\n' ' ')
  passed=$(echo "$out" | grep -cE '^test .* ok')
  if [ "$m" = unmutated ]; then
    if [ -z "$failed" ] && [ "$passed" -ge "$MIN_PASS" ]; then echo "unmutated: pass ($passed tests)"; else echo "unmutated: FAILED: $failed"; bad=1; fi
    if r=$(race); then echo "unmutated: racecheck clean"; else echo "$r"; echo "unmutated: racecheck NOT clean"; bad=1; fi
  elif echo "$out" | grep -qE '^mutation|^error(\[|: could not)|could not compile'; then echo "HARNESS: $m did not apply or build"; bad=1
  elif [ -n "$failed" ]; then echo "KILLED $m (tests): $failed"
  elif [ "$passed" -lt "$MIN_PASS" ]; then echo "KILLED $m (tests): only $passed tests completed (crash or hang)"
  elif ! r=$(race); then
    echo "$r" | sed 's/^/  /'
    if echo "$r" | grep -qE '^RACE .* hazards=[1-9][0-9]*'; then echo "KILLED $m (racecheck): $(echo "$r" | grep -E '^RACE .* hazards=[1-9]' | head -1)"
    elif echo "$r" | grep -qE '^RACE .* hazards=0 test=FAILED'; then echo "KILLED $m (tests under racecheck): $(echo "$r" | grep -E 'test=FAILED' | head -1)"
    else echo "HARNESS: $m racecheck gave no hazard count"; bad=1; fi
  else echo "SURVIVED $m: the tests passed on the mutated kernel and racecheck is clean"; echo "$out" | grep -q EQUIVALENT || bad=1; fi
done
rsh "rm -f $MP; cd $D/$MT && cp $D/$BT/$KFILE $KFILE"
exit $bad
