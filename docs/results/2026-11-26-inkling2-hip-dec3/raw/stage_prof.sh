#!/usr/bin/env bash
# Window 1 stage (lane ink-hip-dec3): rocprofv3 kernel trace of main f3a150646's decode (short and 4k prompt). Budget: radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-dec3
S=$REPO/.scratch
OUT=$REPO/logs/w1
rm -rf "$OUT"; mkdir -p "$OUT"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-main
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0"
now() { date '+%F %T'; }
line() { grep -aE "finish=|steps=|Error|panicked" | cut -c1-300; }
for p in short 4k; do
  [ $p = short ] && PR="$PS" || PR="$P4"
  echo "[stage] rocprof $p $(now)"
  mkdir -p "$OUT/rp-$p"
  INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 bash "$S/run.sh" rocprofv3 --kernel-trace --output-format csv -d /work/logs/w1/rp-$p -o run -- $B $G --max-tokens 24 -p "$PR" 2>&1 | line
  csv=$(find "$OUT/rp-$p" -name '*kernel_trace.csv' | head -1)
  echo "kernel trace $p: ${csv:-none}"
  [ -n "$csv" ] && gzip -kf "$csv" && python3 "$S/ktrace.py" "$csv" 8
done
for p in short 4k; do [ $p = short ] && PR="$PS" || PR="$P4"; echo "[stage] main unprofiled $p 128 $(now)"; bash "$S/run.sh" $B $G --max-tokens 128 -p "$PR" 2>&1 | grep -aoE "decode [0-9]+ ms \([0-9.]+ tok/s\)|steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+" | tr "\n" " "; echo; done
echo "[stage] done $(now)"
