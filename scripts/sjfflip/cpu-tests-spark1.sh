#!/usr/bin/env bash
# sjfflip: cpu gates on spark1 (cpu only, no GPU), own src and target dir. Run under:
#   fleetq run --host spark1 --load cpu --mem 15 --expect 45 --label "sjfflip cpu tests" -- scripts/sjfflip/cpu-tests-spark1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L='~/intarsiaml-sjfflip-cpu'; R="$L/src"
RPATH='export PATH=$HOME/.cargo/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes spark1 "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SHA=$(git -C "$REPO" rev-parse HEAD)
rsh "mkdir -p $R && rm -rf $L/stage && mkdir -p $L/stage"
git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes spark1 "tar -x -C $L/stage"
rsh "rsync -a --checksum --delete --exclude target --exclude golden --exclude tools/.venv $L/stage/ $R/ && rm -rf $L/stage"
echo "sha $SHA"
rc=0
rsh "cd $R && CARGO_BUILD_JOBS=4 cargo test --locked -p intarsia-core --lib --tests 2>&1 | grep -E '^(test result|error|failures:|test .*FAILED|     Running)'; exit \${PIPESTATUS[0]}" || rc=1
rsh "cd $R && CARGO_BUILD_JOBS=4 cargo test --locked -p intarsia-cli --test docs_env_vars 2>&1 | grep -E '^(test result|error|failures:|test .*FAILED)'; exit \${PIPESTATUS[0]}" || rc=1
echo "cpu gates rc=$rc"; exit $rc
