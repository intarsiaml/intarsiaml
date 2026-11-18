#!/usr/bin/env bash
# sjfflip: cpu gates. Run under: fleetq run --host endor --load cpu --expect 40 --label "sjfflip cpu tests" -- scripts/sjfflip/cpu-tests.sh
set -uo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target}"
rc=0
cargo test -p intarsia-core --lib --tests 2>&1 | grep -E '^(test result|error|failures:|test .*FAILED|     Running)' ; [[ ${PIPESTATUS[0]} == 0 ]] || rc=1
cargo test -p intarsia-cli --test docs_env_vars 2>&1 | grep -E '^(test result|error|failures:|test .*FAILED)'; [[ ${PIPESTATUS[0]} == 0 ]] || rc=1
echo "cpu gates rc=$rc"; exit $rc
