#!/bin/zsh
# order experiment: grid order kvh-major (default) vs row-major (INTARSIA_METAL_ROWS_ORDER=slow); attention only
set -u
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot"; exit 4; }
cd /opt/code/intarsiaml-g0b
O=docs/results/raw/2026-10-15-metal-g0b-attn/order
mkdir -p $O
INTARSIA_METAL_ROWS_ORDER=slow cargo test -p intarsia-backend-metal --release --test g0b_attn -- --test-threads=1 > $O/test-slow.log 2>&1; echo "test slow exit $?" | tee $O/slot.log
/Users/josef/bin/macmon pipe -i 100 > $O/macmon.jsonl 2>/dev/null &
MM=$!
sleep 2
for ord in default slow; do
  if [ $ord = slow ]; then export INTARSIA_METAL_ROWS_ORDER=slow; else unset INTARSIA_METAL_ROWS_ORDER; fi
  G0B_OUT=$O/g0b-$ord.csv G0B_ROUNDS=10 ./target/release/examples/g0b_bench attn 1024 4096 16384 > $O/g0b-$ord.log 2>&1
  echo "bench $ord exit $?" | tee -a $O/slot.log
done
kill $MM 2>/dev/null
echo "end" | tee -a $O/slot.log
