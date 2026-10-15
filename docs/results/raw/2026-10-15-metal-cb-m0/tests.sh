#!/bin/zsh
# usage: tests.sh <outdir>   (inside an exclusive endor slot): the M0 test gates
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
O=$1; mkdir -p $O
cd /opt/code/intarsiaml-cb-m0
cargo test --workspace --release > $O/cargo-test-workspace.log 2>&1; echo "cargo test --workspace --release: exit $?"
cargo test -p intarsia-backend-metal --release > $O/cargo-test-metal.log 2>&1; echo "cargo test -p intarsia-backend-metal --release: exit $?"
cargo build --release -p intarsia-cli --features metal > $O/build-cli.log 2>&1; echo "build cli: exit $?"
./target/release/intarsia kernels test --backend metal > $O/kernels-test-metal.log 2>&1; echo "kernels test --backend metal: exit $?"
