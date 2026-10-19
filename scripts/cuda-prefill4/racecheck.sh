#!/usr/bin/env bash
# Shared-memory race check of gdn_delta_rule_bf16_split with compute-sanitizer (racecheck), for the mutant the bitwise tests
# cannot see: a missing __syncwarp lets the double buffer race, but lanes of a warp that stay converged never show it.
#   fleetq run --host spark1 --mem 12 --label "..." -- scripts/cuda-prefill4/racecheck.sh [mutation ...]
# In ~/intarsia-decfuse/p4-mut (made by mutcheck.sh): first the unmutated kernel (must report 0 hazards), then each named
# mutation (default no-loop-sync; KILLED when racecheck reports a hazard). Runs gdn_split_state_carries_across_calls
# (small T) under `compute-sanitizer --tool racecheck`.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
MUTS=${*:-no-loop-sync}
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
D='~/intarsia-decfuse'
rsh "test -d $D/p4-mut || { mkdir -p $D/p4-mut && rsync -a --delete $D/p4-dev/ $D/p4-mut/; }" > /dev/null
scp -q "$REPO/scripts/cuda-prefill4/mutate.py" "$HOST:/tmp/p4_race_mutate_$FLEETQ_JOB.py"
bad=0
for m in unmutated $MUTS; do
  out=$(rsh "cd $D/p4-mut && cp $D/p4-dev/kernels/cuda/gdn.cu kernels/cuda/ && { [ $m = unmutated ] || python3 /tmp/p4_race_mutate_$FLEETQ_JOB.py . $m; } && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --test gdn_split --no-run 2>&1 | grep -E '^(applied|mutation|error)|could not compile'; BIN=\$(ls -t target/acc/release/deps/gdn_split-* | grep -v '\\.d\$' | head -1); INTARSIA_REQUIRE_GPU=1 timeout 900 compute-sanitizer --tool racecheck --racecheck-report all \$BIN --test-threads=1 gdn_split_state_carries_across_calls 2>&1 | grep -E 'RACECHECK SUMMARY|hazard|Error|ERROR|test result' | sort | uniq -c | head -12")
  echo "--- $m"; echo "$out"
  hz=$(echo "$out" | grep -E 'RACECHECK SUMMARY' | grep -Eo '[0-9]+ hazard' | grep -Eo '[0-9]+' | head -1)
  [ -z "$hz" ] && hz=$(echo "$out" | grep -cE 'hazard')
  if [ "$m" = unmutated ]; then
    if [ "$hz" = 0 ]; then echo "unmutated: racecheck clean"; else echo "unmutated: racecheck reports $hz hazard(s)"; bad=1; fi
  elif [ "$hz" != 0 ]; then echo "KILLED $m: racecheck reports $hz hazard(s)"
  else echo "SURVIVED $m: racecheck clean"; bad=1; fi
done
rsh "rm -f /tmp/p4_race_mutate_$FLEETQ_JOB.py; cd $D/p4-mut && cp $D/p4-dev/kernels/cuda/gdn.cu kernels/cuda/"
exit $bad
