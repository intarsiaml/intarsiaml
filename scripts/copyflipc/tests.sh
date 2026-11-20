#!/bin/zsh
# copyflipc test gate, inside a fleetq slot on endor (--load cpu). Own target dir inside the worktree.
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; cd $W; export CARGO_TARGET_DIR=$W/target-cfc-test
cargo test -p intarsia-core -p intarsia-models -p intarsia-cli --lib --tests 2>&1 | tee docs/results/2026-11-20-copyflipc/raw/tests-full.log | grep -aE "^(test result|error|running|failures:)|FAILED|panicked|Running"
echo "exit=${pipestatus[1]}"
