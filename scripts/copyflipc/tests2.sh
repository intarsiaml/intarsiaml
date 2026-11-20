#!/bin/zsh
# post-merge recheck (docs_env_vars sees the final tuning.md): fleetq slot on endor, --load cpu
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; cd $W; export CARGO_TARGET_DIR=$W/target-cfc-test
cargo test -p intarsia-cli --test docs_env_vars 2>&1 | grep -aE "^test result|error|FAILED"
cargo test -p intarsia-core --lib --test copy_default_unset --test copy_default_zero --test copy_default_empty --test copy_default_explicit 2>&1 | grep -aE "^test result|error|FAILED"
cargo test -p intarsia-models --lib 2>&1 | grep -aE "^test result|error|FAILED"
echo done
