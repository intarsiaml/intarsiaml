#!/bin/zsh
# Second slot run: split-pass variants (register-pressure experiment) for R = 6, 8, 16 against the single-pass R = 4..8 and the serving kernel.
set -u
cd /opt/code/intarsiaml-g0
OUT=docs/results/raw/2026-10-15-metal-g0-multirow/split
mkdir -p $OUT
echo "start $(date)" | tee $OUT/slot.log
cargo test -p intarsia-backend-metal --release --test multirow > $OUT/cargo-test-multirow.log 2>&1; echo "cargo test multirow exit $?" | tee -a $OUT/slot.log
sleep 15
/Users/josef/bin/macmon pipe -i 100 > $OUT/macmon.jsonl 2>/dev/null &
MM=$!
sleep 2
for run in 1 2; do
  G0_ONLY_SPLIT=1 G0_OUT=$OUT/g0-aligned-run$run.csv ./target/release/examples/gemv_g0 > $OUT/g0-aligned-run$run.log 2>&1; echo "bench aligned run$run exit $?" | tee -a $OUT/slot.log
done
G0_ONLY_SPLIT=1 G0_A8=1 G0_OUT=$OUT/g0-a8.csv ./target/release/examples/gemv_g0 > $OUT/g0-a8.log 2>&1; echo "bench a8 exit $?" | tee -a $OUT/slot.log
sleep 2
kill $MM 2>/dev/null
echo "end $(date)" | tee -a $OUT/slot.log
