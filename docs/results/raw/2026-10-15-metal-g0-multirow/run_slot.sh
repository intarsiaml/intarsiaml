#!/bin/zsh
# G0 slot script: run inside an exclusive fleetq slot on endor (FLEETQ_JOB set by the caller).
# order: correctness (tests, kernels test) first, then the benches with macmon logging at 100 ms.
set -u
cd /opt/code/intarsiaml-g0
OUT=docs/results/raw/2026-10-15-metal-g0-multirow
echo "start $(date)" | tee $OUT/slot.log
cargo test -p intarsia-backend-metal --release > $OUT/cargo-test-metal.log 2>&1; echo "cargo test exit $?" | tee -a $OUT/slot.log
./target/release/intarsia kernels test --backend metal -m /Users/josef/models/Qwen3-8B > $OUT/kernels-test-metal-default.log 2>&1; echo "kernels test (default) exit $?" | tee -a $OUT/slot.log
INTARSIA_METAL_MULTIROW=1 ./target/release/intarsia kernels test --backend metal -m /Users/josef/models/Qwen3-8B > $OUT/kernels-test-metal-multirow.log 2>&1; echo "kernels test (MULTIROW=1) exit $?" | tee -a $OUT/slot.log
INTARSIA_METAL_MULTIROW=1 ./target/release/intarsia kernels test --backend metal -m /Users/josef/models/Qwen3-8B --json > $OUT/kernels-test-metal-multirow.json 2>/dev/null
sleep 20
/Users/josef/bin/macmon pipe -i 100 > $OUT/macmon.jsonl 2>/dev/null &
MM=$!
sleep 2
for run in 1 2; do
  G0_OUT=$OUT/g0-aligned-run$run.csv ./target/release/examples/gemv_g0 > $OUT/g0-aligned-run$run.log 2>&1; echo "bench aligned run$run exit $?" | tee -a $OUT/slot.log
done
G0_A8=1 G0_OUT=$OUT/g0-a8.csv ./target/release/examples/gemv_g0 > $OUT/g0-a8.log 2>&1; echo "bench a8 exit $?" | tee -a $OUT/slot.log
sleep 2
kill $MM 2>/dev/null
echo "end $(date)" | tee -a $OUT/slot.log
