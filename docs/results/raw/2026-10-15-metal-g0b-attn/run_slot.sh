#!/bin/zsh
# G0b slot script: runs INSIDE an exclusive fleetq slot on endor (FLEETQ_JOB set by slot.sh).
# stages: bitwise tests (abort on failure), attn_hash vs the B5 reference, full cargo test, kernels test, macmon + benches, align_copy.
set -u
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -n "$DEVELOPER_DIR" ] && [ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR empty or missing"; exit 3; }
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot"; exit 4; }
cd /opt/code/intarsiaml-g0b
OUT=docs/results/raw/2026-10-15-metal-g0b-attn
TAG=${1:-run}
mkdir -p $OUT/$TAG
O=$OUT/$TAG
echo "start $(date) job $FLEETQ_JOB DEVELOPER_DIR=$DEVELOPER_DIR" | tee $O/slot.log
cargo test -p intarsia-backend-metal --release --test g0b_attn --test g0b_small -- --nocapture --test-threads=1 > $O/cargo-test-g0b.log 2>&1
rc=$?; echo "g0b tests exit $rc $(date)" | tee -a $O/slot.log
[ $rc -eq 0 ] || exit $rc
./target/release/examples/attn_hash > $O/attn_hash.txt 2>&1
diff <(grep -v '^$' docs/results/raw/2026-10-14-metal-phase-b/b5/attn_hash-new3.txt) <(grep -v '^$' $O/attn_hash.txt) > $O/attn_hash.diff 2>&1
echo "attn_hash vs b5 reference: diff exit $? $(date)" | tee -a $O/slot.log
cargo test -p intarsia-backend-metal --release > $O/cargo-test-metal.log 2>&1
echo "cargo test metal exit $? $(date)" | tee -a $O/slot.log
./target/release/intarsia kernels test --backend metal -m /Users/josef/models/Qwen3-8B > $O/kernels-test-metal.log 2>&1
echo "kernels test exit $? $(date)" | tee -a $O/slot.log
/Users/josef/bin/macmon pipe -i 100 > $O/macmon.jsonl 2>/dev/null &
MM=$!
sleep 2
G0B_OUT=$O/g0b.csv G0B_ROUNDS=${G0B_ROUNDS:-20} ./target/release/examples/g0b_bench > $O/g0b.log 2>&1
echo "bench exit $? $(date)" | tee -a $O/slot.log
sleep 1
kill $MM 2>/dev/null
./target/release/examples/align_copy > $O/align_copy.log 2>&1
echo "align_copy exit $? $(date)" | tee -a $O/slot.log
echo "end $(date)" | tee -a $O/slot.log
