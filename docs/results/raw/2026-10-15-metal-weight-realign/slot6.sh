#!/bin/zsh
# slot 6: tests with re-alignment on by default (INTARSIA_METAL_REALIGN=1) after padding the copy to a 16-byte multiple,
# then the plain metal-crate suite once more on the final tree.
W=/opt/code/intarsiaml-weight-realign
R=$W/docs/results/raw/2026-10-15-metal-weight-realign/slot6
mkdir -p $R
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 8 --label "metal-weight-realign: slot6 env-on tests + final metal suite" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
git rev-parse HEAD
INTARSIA_METAL_REALIGN=1 cargo test -p intarsia-backend-metal --release --test realign --test wrap_host --test fill --test multirow --test gemv > $R/realign-env.log 2>&1; echo "REALIGN=1 exit $?"
cargo test -p intarsia-backend-metal --release --no-fail-fast > $R/cargo-test-metal.log 2>&1; echo "metal suite exit $?"
echo done
