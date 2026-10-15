#!/bin/zsh
# cb-m1 step 2 gates 1 and 2, one exclusive endor slot: usage gates.sh <outdir> <rev>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export INTARSIA_TEST_REAL_MODEL=1
O=$1; REV=$2; mkdir -p $O
HERE=${0:A:h}
S=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad
cd /opt/code/intarsiaml-cb-wire
echo "HEAD $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) binary $(shasum $S/bin/intarsia-cbwire-current)"
cargo test --workspace --release > $O/cargo-test-workspace.log 2>&1; echo "cargo test --workspace --release: exit $?"
cargo test -p intarsia-backend-metal --release > $O/cargo-test-metal.log 2>&1; echo "cargo test -p intarsia-backend-metal --release: exit $?"
cmp -s ./target/release/intarsia $S/bin/intarsia-cbwire-current && echo "cli binary == identity binary"
$HERE/run_id.sh $O/id-head intarsia-cbwire-current base off bon4 on4 > $O/id-head.log 2>&1; echo "identity: exit $?"
