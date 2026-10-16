#!/bin/bash
# M4/M5 profile slot: B=8 decode step breakdown, plain / cb / kernel trace, 1k and 4k context. Runs inside a fleetq slot.
# usage: run.sh BIN_DIR OUT_DIR
set -u
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
BIN=$1; OUT=$2; mkdir -p "$OUT"
macmon pipe -i 500 > "$OUT/macmon.jsonl" 2>/dev/null &
MM=$!
trap 'kill $MM 2>/dev/null' EXIT
date -u > "$OUT/start.txt"; sw_vers > "$OUT/versions.txt"
for ctx in 1024 4096; do
  # plain wall time, B=1 and B=8, three repeats (ABAB-ish by interleaving)
  for rep in 1 2 3; do
    for b in 1 8; do
      "$BIN/batch_prof" --ctx $ctx --b $b --steps 60 --warm 10 --marks "$OUT/plain-c$ctx-b$b-r$rep.marks" > "$OUT/plain-c$ctx-b$b-r$rep.log" 2>&1
      grep -q median "$OUT/plain-c$ctx-b$b-r$rep.log" || { echo "first run failed" >&2; exit 1; }
    done
  done
  for b in 1 8; do
    for mode in cb kernel; do
      INTARSIA_METAL_PROFILE=$mode:"$OUT/$mode-c$ctx-b$b.jsonl" "$BIN/batch_prof" --ctx $ctx --b $b --steps 24 --warm 6 --marks "$OUT/$mode-c$ctx-b$b.marks" > "$OUT/$mode-c$ctx-b$b.log" 2>&1
    done
  done
  "$BIN/batch_step" --ctx $ctx --rounds 8 --steps 16 --batches 1,2,4,8 > "$OUT/batch_step-c$ctx.log" 2>&1
done
date -u > "$OUT/end.txt"
