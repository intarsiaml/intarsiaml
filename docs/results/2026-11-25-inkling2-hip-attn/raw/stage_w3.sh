#!/usr/bin/env bash
# GPU stage, lane ink-hip-attn window 3 (op 75 v2 + flash prefill). Prebuilt binaries only. Budget: radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-attn
R=$REPO/docs/results/2026-11-25-inkling2-hip-attn/raw
T="cargo test --release -j8 -p intarsia-backend-hip --test ink_attn"
M=/models/Inkling-2-Mini
I=/target/release/intarsia
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0"
line() { grep -aoE "finish=.*|Error.*|panicked.*" | cut -c1-300; }
now() { date '+%F %T'; }
echo "[stage] admission (full) $(now)"
bash "$R/run.sh" $T -- --exact admission_full_sweep --nocapture 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|test result|panicked| fail " | head -40
echo "[stage] rows-exact $(now)"
bash "$R/run.sh" $T -- --exact rows_exact_real_shapes --nocapture 2>&1 | grep -aE "rows-exact|test result|panicked" | head
for sk in 0 128 256; do
  echo "[stage] bench decode split_keys=$sk $(now)"
  bash "$R/run.sh" env INTARSIA_INK_SPLIT_KEYS=$sk $T -- --exact bench_decode --ignored --nocapture 2>&1 | grep -aE "BENCH|panicked"
done
for dg in 1 2 3; do
  echo "[stage] bench decode diag=$dg (wrong output, timing only) $(now)"
  bash "$R/run.sh" env INTARSIA_INK_DEC_DIAG=$dg $T -- --exact bench_decode --ignored --nocapture 2>&1 | grep -aE "BENCH|panicked"
done
echo "[stage] bench prefill $(now)"
bash "$R/run.sh" $T -- --exact bench_prefill --ignored --nocapture 2>&1 | grep -aE "BENCH|panicked"
for g in short long; do
  echo "[stage] G2 $g (flash prefill) $(now)"
  bash "$R/run.sh" $I accuracy -m $M --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors 2>&1 | grep -aE "floor verdict|prefill floor|Error|panicked" | cut -c1-400
done
for f in 0 1; do
  for p in short 4k; do
    [ $p = short ] && PR="$PS" || PR="$P4"
    echo "[stage] model decode fused=$f $p 256 $(now)"
    bash "$R/run.sh" env INTARSIA_INK_ATTN_FUSED=$f $I $G --max-tokens 256 -p "$PR" 2>&1 | line
  done
done
echo "[stage] done $(now)"
