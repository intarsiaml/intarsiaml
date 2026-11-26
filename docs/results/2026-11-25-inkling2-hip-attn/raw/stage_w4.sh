#!/usr/bin/env bash
# GPU stage, lane ink-hip-attn window 4 (op 75 rule + flash knob). Prebuilt binaries only. Budget: radiance down <= 5 min.
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
for fl in 0 1; do
  echo "[stage] admission (full) flash=$fl $(now)"
  bash "$R/run.sh" env INTARSIA_HIP_INK_ATTN_FLASH=$fl $T -- --exact admission_full_sweep --nocapture 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|test result|panicked| fail " | head -20
done
echo "[stage] rows-exact $(now)"
bash "$R/run.sh" $T -- --exact rows_exact_real_shapes --nocapture 2>&1 | grep -aE "rows-exact|test result|panicked" | head
for fk in 2048 0 100000000; do
  echo "[stage] bench decode INTARSIA_INK_DEC_FUSED_KEYS=$fk $(now)"
  bash "$R/run.sh" env INTARSIA_INK_DEC_FUSED_KEYS=$fk $T -- --exact bench_decode --ignored --nocapture 2>&1 | grep -aE "BENCH|panicked"
done
echo "[stage] bench prefill $(now)"
bash "$R/run.sh" $T -- --exact bench_prefill --ignored --nocapture 2>&1 | grep -aE "BENCH|panicked"
for fl in 0 1; do
  for g in short long; do
    echo "[stage] G2 $g flash=$fl $(now)"
    bash "$R/run.sh" env INTARSIA_HIP_INK_ATTN_FLASH=$fl $I accuracy -m $M --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors 2>&1 | grep -aE "prefill floor|floor verdict|Error|panicked" | cut -c1-400
  done
done
for f in 0 1; do
  for p in short 4k; do
    [ $p = short ] && PR="$PS" || PR="$P4"
    echo "[stage] model fused=$f (1 = op 75 with the rule) flash=0 $p 256 $(now)"
    bash "$R/run.sh" env INTARSIA_INK_ATTN_FUSED=$f INTARSIA_HIP_INK_ATTN_FLASH=0 $I $G --max-tokens 256 -p "$PR" 2>&1 | line
  done
done
echo "[stage] model fused=1 flash=1 4k 256 $(now)"
bash "$R/run.sh" env INTARSIA_INK_ATTN_FUSED=1 INTARSIA_HIP_INK_ATTN_FLASH=1 $I $G --max-tokens 256 -p "$P4" 2>&1 | line
echo "[stage] done $(now)"
