#!/usr/bin/env bash
# Shared-memory race check of the CUDA kernels listed in scripts/cuda-checks/race-targets.txt, on spark1, inside the
# caller's fleetq slot (FLEETQ_JOB goes on every ssh):
#   racecheck.sh <tree> [<test binary>:<test name> ...]
# <tree> is ~/intarsia-decfuse/<tree> (synced and built by scripts/cuda-decode/build.sh). Without targets, every line of
# race-targets.txt (KFILE=<kernel source> keeps only that kernel's lines). Per target: build the test binary (no-op when
# cargo test already built it), run the one test under `compute-sanitizer --tool racecheck --racecheck-report all`, print
#   RACE <bin>:<test> hazards=<N> test=<ok|FAILED|missing>
# Exit 0 only if every target reports 0 hazards and its test passes; 1 otherwise (2: no target).
# Called by scripts/cuda-decode/gpu-test.sh (the CUDA test path; RACECHECK=0 skips) and by the mutation harness
# scripts/cuda-prefill4/mutcheck.sh for every mutant that passes the bitwise tests.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TREE=${1:?tree}; shift
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
D='~/intarsia-decfuse'
if (( $# > 0 )); then TARGETS="$*"; else
  TARGETS=$(grep -vE '^[[:space:]]*(#|$)' "$REPO/scripts/cuda-checks/race-targets.txt" | awk -v k="${KFILE:-}" 'k == "" || $3 == k {print $1":"$2}' | tr '\n' ' ')
fi
[[ -n "${TARGETS// /}" ]] || { echo "RACE: no target"; exit 2; }
bad=0
for t in $TARGETS; do
  bin=${t%%:*}; test=${t#*:}
  out=$(rsh "cd $D/$TREE && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --test $bin --no-run 2>&1 | grep -E '^error|could not compile'; B=\$(ls -t target/acc/release/deps/$bin-* 2>/dev/null | grep -v '\\.d\$' | head -1); [ -n \"\$B\" ] || { echo 'no test binary'; exit 0; }; F=/tmp/racecheck_\$\$.log; INTARSIA_REQUIRE_GPU=1 timeout 900 compute-sanitizer --tool racecheck --racecheck-report all \$B --test-threads=1 --exact $test > \$F 2>&1; grep -E 'RACECHECK SUMMARY' \$F | tail -1; grep -Eo 'test result: [a-zA-Z]+\\. [0-9]+ passed; [0-9]+ failed' \$F | tail -1; echo \"hazard lines: \$(grep -c 'hazard detected' \$F)\"; grep -q 'RACECHECK SUMMARY' \$F || { echo '(no summary; log tail:)'; tail -3 \$F | cut -c1-200; }; grep -E 'hazard detected|ERROR' \$F | sed -E 's/ in block \\([0-9,]+\\)//; s/0x[0-9a-f]+/ADDR/g' | sort | uniq -c | sort -rn | head -${RACE_LINES:-6}; rm -f \$F")
  echo "$out" | sed 's/^/    /'
  hz=$(echo "$out" | grep -E 'RACECHECK SUMMARY' | grep -Eo '[0-9]+ hazard' | grep -Eo '[0-9]+' | head -1)
  # no summary (the sanitizer stopped early): the hazard lines it printed still count; 0 lines stays "unknown"
  if [[ -z "$hz" ]]; then n=$(echo "$out" | grep -Eo 'hazard lines: [0-9]+' | grep -Eo '[0-9]+$'); [[ -n "$n" && "$n" != 0 ]] && hz=$n; fi
  if echo "$out" | grep -qE 'test result: ok\. 1 passed'; then res=ok
  elif echo "$out" | grep -qE 'test result'; then res=FAILED
  else res=missing; fi
  [[ -z "$hz" ]] && hz=unknown
  echo "RACE $bin:$test hazards=$hz test=$res"
  [[ "$hz" == 0 && "$res" == ok ]] || bad=1
done
exit $bad
