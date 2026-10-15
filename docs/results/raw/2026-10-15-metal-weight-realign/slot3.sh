#!/bin/zsh
# metal-weight-realign slot 3: final test runs after the multirow.rs merge fixup (slot 1's runs stopped at the stale
# `serving_selection_unchanged`). One exclusive fleetq slot.
W=/opt/code/intarsiaml-weight-realign
R=$W/docs/results/raw/2026-10-15-metal-weight-realign/slot3
mkdir -p $R
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 15 --label "metal-weight-realign: slot3 final tests" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
git rev-parse HEAD
cargo test -p intarsia-backend-metal --release --no-fail-fast > $R/cargo-test-metal.log 2>&1; echo "cargo test metal exit $?"
cargo test --workspace --release --no-fail-fast > $R/cargo-test-workspace.log 2>&1; echo "cargo test workspace exit $?"
INTARSIA_METAL_MULTIROW=1 cargo test -p intarsia-backend-metal --release --test multirow > $R/cargo-test-multirow-env.log 2>&1; echo "multirow with MULTIROW=1 exit $?"
echo done
