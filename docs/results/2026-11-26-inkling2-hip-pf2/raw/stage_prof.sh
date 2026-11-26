#!/usr/bin/env bash
# Window 1 stage (lane ink-hip-pf2): rocprofv3 kernel trace of main 87c9745c0's prefill at 4k and 16k with the new prefill
# settings (INTARSIA_HIP_INK_MOE_WMMA=1 INTARSIA_HIP_INK_ATTN_FLASH=1 --prefill-chunk 2048), then one unprofiled run of each.
# Budget: radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-pf2
S=$REPO/.scratch
OUT=$REPO/logs/w1
rm -rf "$OUT"; mkdir -p "$OUT"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-${BIN:-main}
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
P16="$P4
$P4
$P4
$P4"
export INTARSIA_HIP_INK_MOE_WMMA=1 INTARSIA_HIP_INK_ATTN_FLASH=1
G="generate -m $M --backend hip --max-seq 20480 --spec off --ignore-eos --profile-every 0 --prefill-chunk 2048"
now() { date '+%F %T'; }
line() { grep -aE "finish=|prefill [0-9]+ ms|Error|panicked" | cut -c1-300; }
for p in 4k 16k; do
  [ $p = 4k ] && PR="$P4" || PR="$P16"
  echo "[stage] rocprof $p $(now)"
  mkdir -p "$OUT/rp-$p"
  INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 bash "$S/run.sh" rocprofv3 --kernel-trace --output-format csv -d /work/logs/w1/rp-$p -o run -- $B $G --max-tokens 2 -p "$PR" 2>&1 | line
  csv=$(find "$OUT/rp-$p" -name '*kernel_trace.csv' | head -1)
  echo "kernel trace $p: ${csv:-none} $(wc -l < "$csv" 2>/dev/null) lines"
done
for p in 4k 16k; do
  [ $p = 4k ] && PR="$P4" || PR="$P16"
  echo "[stage] unprofiled $p $(now)"
  bash "$S/run.sh" $B $G --max-tokens 2 -p "$PR" 2>&1 | line
done
echo "[stage] done $(now)"
