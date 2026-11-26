#!/usr/bin/env bash
# GPU stage, lane ink-hip-attn window 2 (op 75): admission 70/71/75 (quick, then full), rows-exact (pair vs op 75), kernel bench,
# whole-model decode with INTARSIA_INK_ATTN_FUSED=0 (the pair) vs 1 (op 75). Prebuilt binaries only. Budget: radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-attn
R=$REPO/docs/results/2026-11-25-inkling2-hip-attn/raw
T="cargo test --release -j8 -p intarsia-backend-hip --test ink_attn"
M=/models/Inkling-2-Mini
I=/target/release/intarsia
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0"
line() { grep -aE "finish=|steps=|Error|error|panicked" | grep -aoE "finish=.*|steps=.*|Error.*|error.*|panicked.*" | cut -c1-300; }
now() { date '+%F %T'; }
echo "[stage] admission (full) $(now)"
bash "$R/run.sh" $T -- --exact admission_full_sweep --nocapture
echo "[stage] rows-exact $(now)"
bash "$R/run.sh" $T -- --exact rows_exact_real_shapes --nocapture
echo "[stage] bench $(now)"
bash "$R/run.sh" $T -- --exact bench_decode --ignored --nocapture
for f in 0 1 0 1; do
  echo "[stage] model decode fused=$f short 256 $(now)"
  bash "$R/run.sh" env INTARSIA_INK_ATTN_FUSED=$f $I $G --max-tokens 256 -p "$PS" 2>&1 | line
done
for f in 0 1; do
  echo "[stage] model decode fused=$f 4k 256 $(now)"
  bash "$R/run.sh" env INTARSIA_INK_ATTN_FUSED=$f $I $G --max-tokens 256 -p "$P4" 2>&1 | line
done
echo "[stage] done $(now)"
