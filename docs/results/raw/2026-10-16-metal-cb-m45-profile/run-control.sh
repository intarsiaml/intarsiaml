#!/bin/bash
# Control slot: today's batch_step (66feda6) vs the earlier scratchpad binary batch_step-v3 (Oct 15 16:58, M1 step 1 era,
# provenance not verified), same slot, alternating, with the GPU clock logged. usage: run-control.sh BIN_DIR OUT_DIR
set -u
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
BIN=$1; OUT=$2; mkdir -p "$OUT"
macmon pipe -i 500 > "$OUT/macmon-control.jsonl" 2>/dev/null &
MM=$!
trap 'kill $MM 2>/dev/null' EXIT
for rep in 1 2 3; do
  for v in batch_step batch_step-ctl; do
    date +%s > "$OUT/control-$v-r$rep.t0"
    "$BIN/$v" --ctx 1024 --rounds 6 --steps 16 --batches 1,8 > "$OUT/control-$v-r$rep.log" 2>&1
    date +%s > "$OUT/control-$v-r$rep.t1"
  done
done
