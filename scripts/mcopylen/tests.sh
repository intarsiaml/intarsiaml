#!/bin/zsh
# mcopylen: the unit tests that cover the changed path (cpu): spec rule tests, the engine tests with copy chains (adaptive cases included),
# the docs drift test.
#   fleetq run --host endor --load cpu --mem 20 --expect 25 --label "mcopylen: tests" -- scripts/mcopylen/tests.sh
W=${0:A:h:h:h}; export CARGO_TARGET_DIR=$W/target; cd $W
cargo test --release -p intarsia-core --lib spec:: 2>&1 | grep -v "^ *Compiling" | tail -15
cargo test --release -p intarsia-models --test mtp -- --nocapture copy 2>&1 | grep -v "^ *Compiling" | tail -30
cargo test --release -p intarsia-cli --test docs_env_vars 2>&1 | tail -8
