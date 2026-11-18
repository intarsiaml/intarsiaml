#!/bin/zsh
# herdflip: knob tests + the crates the default touches + the env-var table check. Inside a fleetq slot on endor (--load cpu).
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; RAW=$W/docs/results/2026-11-18-herdflip/raw; mkdir -p $RAW
export CARGO_TARGET_DIR=$W/target-test
cd $W
cargo test -p intarsia-core > $RAW/test-core.log 2>&1; echo "intarsia-core: exit $?"
cargo test -p intarsia-models > $RAW/test-models.log 2>&1; echo "intarsia-models: exit $?"
cargo test -p intarsia-cli --test docs_env_vars > $RAW/test-docs-env.log 2>&1; echo "docs_env_vars: exit $?"
grep -h "^test result" $RAW/test-core.log $RAW/test-models.log $RAW/test-docs-env.log | sort | uniq -c | sort -rn | head
