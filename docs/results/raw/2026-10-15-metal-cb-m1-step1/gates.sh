#!/bin/zsh
# cb-m1 step 1 gates, one exclusive endor slot: usage gates.sh <outdir>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
O=$1; mkdir -p $O
HERE=${0:A:h}
S=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad
cd /opt/code/intarsiaml-cb-m1
cargo test --workspace --release > $O/cargo-test-workspace.log 2>&1; echo "cargo test --workspace --release: exit $?"
cargo test -p intarsia-backend-metal --release > $O/cargo-test-metal.log 2>&1; echo "cargo test -p intarsia-backend-metal --release: exit $?"
cargo build --release -p intarsia-cli --features metal > $O/build-cli.log 2>&1; echo "build cli: exit $?"
cmp -s ./target/release/intarsia $S/bin/intarsia-head && echo "cli binary == identity binary"
./target/release/intarsia kernels test --backend metal > $O/kernels-test-metal.log 2>&1; echo "kernels test --backend metal: exit $?"
for c in 128 1024 4096; do
  $S/bin/batch_step-v2 --ctx $c > $O/batch-step-$c.log 2>&1; echo "batch_step ctx $c: exit $?"
done
$HERE/run_id.sh $O/id-head intarsia-head base off bon4 on4 > $O/id-head.log 2>&1; echo "identity: exit $?"
