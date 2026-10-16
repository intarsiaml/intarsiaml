#!/bin/bash
# cb-mode trace with every dispatch its own command buffer, committed back to back and NOT waited on
# (INTARSIA_METAL_COMMIT_EVERY=1): per-launch GPU timestamps without the kernel-mode host wait, so the GPU
# stays continuously loaded like in a normal step. usage: run-ce1.sh BIN_DIR OUT_DIR
set -u
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
BIN=$1; OUT=$2; mkdir -p "$OUT"
macmon pipe -i 250 > "$OUT/macmon-ce1.jsonl" 2>/dev/null &
MM=$!
trap 'kill $MM 2>/dev/null' EXIT
for ctx in 1024 4096; do
  for b in 8 1; do
    INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:"$OUT/ce1-c$ctx-b$b.jsonl" "$BIN/batch_prof" --ctx $ctx --b $b --steps 40 --warm 8 --marks "$OUT/ce1-c$ctx-b$b.marks" > "$OUT/ce1-c$ctx-b$b.log" 2>&1
  done
done
