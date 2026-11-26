#!/usr/bin/env bash
# Window 1 stage (lane ink-hip-dec2): rocprofv3 kernel trace of main's decode (short and 4k prompt), then a quick decode A/B of
# the existing fused router (INTARSIA_INK_ROUTE_FUSED=1 vs unset) on main's binary. Budget: radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-dec2
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
for rep in 1 2; do
  for f in unset 1; do
    for p in short 4k; do
      [ $p = short ] && PR="$PS" || PR="$P4"
      echo "[stage] route_fused=$f $p rep $rep $(now)"
      if [ $f = unset ]; then bash "$S/run.sh" $B $G --max-tokens 128 -p "$PR" 2>&1 | line
      else bash "$S/run.sh" env INTARSIA_INK_ROUTE_FUSED=1 $B $G --max-tokens 128 -p "$PR" 2>&1 | line; fi
    done
  done
done
echo "[stage] done $(now)"
