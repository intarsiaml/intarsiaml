#!/usr/bin/env bash
# endor (Mac), inside `fleetq run --host endor --load cpu` (lane ink-hip-specflip, gate 1): the full cpu suite of the head: intarsia-core
# lib + its scheduler test, every intarsia-models test (lib + tests; real-model / Metal ones skip without INTARSIA_TEST_REAL_MODEL), every
# intarsia-cli test. No INTARSIA_* variable set. Release build, target/ of the worktree. Logs: .scratch/cpu-<label>/.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-hip-specflip
L=$W/.scratch/cpu-${1:-$(date +%H%M)}; mkdir -p "$L"
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset "$v"; done
export CARGO_TARGET_DIR=$W/target
cd "$W"
echo "[cpu] head $(git rev-parse --short HEAD) $(date +%T)"
rc=0
run() {  # name, cargo test args
  local n=$1; shift
  cargo test --release --locked "$@" > "$L/$n.txt" 2>&1; local r=$?
  [ $r -ne 0 ] && rc=1
  echo "[cpu] $n rc $r | passed $(grep -aE '^test result' "$L/$n.txt" | awk '{p+=$4; f+=$6} END {print p" failed "f}') | $(grep -aE '^test .* FAILED|panicked at' "$L/$n.txt" | head -3 | tr '\n' ' ')"
}
run core-lib -p intarsia-core --lib
run core-scheduler -p intarsia-core --test scheduler
run models -p intarsia-models
run cli -p intarsia-cli
echo "[cpu] done $(date +%T) exit $rc"
exit $rc
