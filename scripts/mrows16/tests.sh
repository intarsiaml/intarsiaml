#!/bin/zsh
# mrows16: the unit tests that cover the changed path (Metal rows attention, 9..=16 rows), the cpu spec tests and the docs drift test.
#   fleetq run --host endor --mem 20 --expect 25 --label "mrows16: tests" -- scripts/mrows16/tests.sh
W=${0:A:h:h:h}; export CARGO_TARGET_DIR=$W/target; cd $W
cargo test --release -p intarsia-backend-metal --lib attn 2>&1 | tail -15
cargo test --release -p intarsia-backend-metal --test g0b_attn --test kv_once --test hs_bitwise -- --nocapture --test-threads 1 2>&1 | grep -v "^ *Compiling" | tail -60
cargo test --release -p intarsia-cli --test docs_env_vars 2>&1 | tail -8
