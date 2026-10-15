#!/bin/zsh
# slot 8: final test runs on the final tree: realign test (stderr shown), metal crate suite, workspace suite.
W=/opt/code/intarsiaml-weight-realign
R=$W/docs/results/raw/2026-10-15-metal-weight-realign/slot8
mkdir -p $R
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 10 --label "metal-weight-realign: slot8 final tests" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
git rev-parse HEAD
cargo test -p intarsia-backend-metal --release --test realign -- --nocapture > $R/realign.log 2>&1; echo "realign exit $?"
INTARSIA_METAL_REALIGN=1 cargo test -p intarsia-backend-metal --release --no-fail-fast > $R/cargo-test-metal-realign-on.log 2>&1; echo "metal suite REALIGN=1 exit $?"
cargo test -p intarsia-backend-metal --release --no-fail-fast > $R/cargo-test-metal.log 2>&1; echo "metal suite exit $?"
cargo test --workspace --release --no-fail-fast > $R/cargo-test-workspace.log 2>&1; echo "workspace exit $?"
echo done
