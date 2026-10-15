#!/bin/zsh
# rows-per-threadgroup experiment: bitwise tests and attention bench for INTARSIA_METAL_ROWS_RTG in 1 2 3 4 8
set -u
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot"; exit 4; }
cd /opt/code/intarsiaml-g0b
O=docs/results/raw/2026-10-15-metal-g0b-attn/rtg
mkdir -p $O
: > $O/slot.log
for r in 2 3 4 8; do
  INTARSIA_METAL_ROWS_RTG=$r cargo test -p intarsia-backend-metal --release --test g0b_attn -- --test-threads=1 > $O/test-rtg$r.log 2>&1; echo "test rtg $r exit $?" | tee -a $O/slot.log
done
/Users/josef/bin/macmon pipe -i 100 > $O/macmon.jsonl 2>/dev/null &
MM=$!
sleep 2
for r in 1 2 3 4; do
  INTARSIA_METAL_ROWS_RTG=$r G0B_OUT=$O/g0b-rtg$r.csv G0B_ROUNDS=10 ./target/release/examples/g0b_bench attn 128 1024 4096 16384 > $O/g0b-rtg$r.log 2>&1
  echo "bench rtg $r exit $?" | tee -a $O/slot.log
done
kill $MM 2>/dev/null
echo "end" | tee -a $O/slot.log
