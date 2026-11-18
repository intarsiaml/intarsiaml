#!/bin/zsh
# mgdnv: one json-xform generate at INTARSIA_SPEC_COPY=20,15, knob 1 then 0, with the counter-patched debug binary (scripts/mgdnv/counter-debug.patch.txt): prints the fused-call histogram. The tok/s of this run is a cold first run, not a measurement.
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
B=/opt/code/intarsiaml/.worktrees/mgdnv/target-mgdnv-dbg/release/intarsia
P=/opt/code/intarsiaml/.worktrees/mgdnv/target-mgdnv/cells/json-xform.txt
for knob in 1 0; do
INTARSIA_METAL_GDN_VERIFY_FUSE=$knob INTARSIA_SPEC_COPY=20,15 $B generate -m $M --backend metal --max-seq 8192 --spec mtp --spec-k 2 --chat --no-think --temperature 0 --profile-every 0 -p "$(cat $P)" --max-tokens 1200 < /dev/null 2>&1 >/dev/null | grep -a "mgdnvdbg\|tok/s" | cut -c1-300
echo "knob=$knob done"
done
