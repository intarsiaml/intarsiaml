#!/bin/zsh
# mlen15 test gates, inside a fleetq slot on endor. A target dir of its own (a shared one with the `--features metal` binary would relink it).
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; cd $W; export CARGO_TARGET_DIR=$W/target-len15-test
fails=0
run() { echo "== $*"; "$@" 2>&1 | grep -aE "^(test result|error|warning: unused|running|failures:)|FAILED|panicked" | sort | uniq -c | sort -rn | head -40; [ ${pipestatus[1]} -eq 0 ] || { echo "FAIL: $*"; fails=$((fails+1)); }; }
run cargo test --release -p intarsia-backend-metal
run cargo test --release -p intarsia-cli --test docs_env_vars
run cargo test --release -p intarsia-core --lib spec
echo "test fails=$fails"
