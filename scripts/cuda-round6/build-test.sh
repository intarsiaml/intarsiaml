#!/usr/bin/env bash
# Sync + build this working tree as spark1:~/intarsia-decfuse/$TREE (default r6-dev) with the CUDA test binaries, then run
# the backend test binaries named in TESTS (default "--test gdn_split") once per env set in TEST_ENVS (";"-separated),
# and optionally GDN timing (GDN=1: scripts/cuda-round6/gdn-ab.sh, ring off / on). Small slot, vLLM up:
#   fleetq run --host spark1 --mem 8 --label "..." -- scripts/cuda-round6/build-test.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
TREE=${TREE:-r6-dev}
"$REPO/scripts/cuda-decode/build.sh" "$TREE" --tests 2>&1 | grep -E "^error|error\[|^warning: unused|Finished|intarsia$" | head -20
IFS=';' read -r -a envs <<< "${TEST_ENVS:-INTARSIA_X=0}"
for e in "${envs[@]}"; do
  echo "== tests ${TESTS:---test gdn_split} with $e"
  ssh -n -o BatchMode=yes spark1 "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsia-decfuse/$TREE && env $e CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda ${TESTS:---test gdn_split} -- --test-threads=1 2>&1 | grep -E '^test |panicked|differs|result:' | tail -30"
done
if [[ "${GDN:-0}" == 1 ]]; then EXP=0 ROUNDS=${ROUNDS:-2} "$S/gdn-ab.sh"; fi
