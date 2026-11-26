#!/usr/bin/env bash
# GPU stage of lane ink-hip-attn's unit window (run by window.sh ON tatooine, radiance down, card 0). Prebuilt binaries only
# (the build job ran `cargo test --release -p intarsia-backend-hip --test ink_attn --no-run`). Order: cheapest first; the full
# admission sweep (f64 cpu reference of the real 2048-row chunks) last, so a cap hit loses only that.
set -uo pipefail
R=$HOME/intarsiaml-ink-hip-attn/docs/results/2026-11-25-inkling2-hip-attn/raw
T="cargo test --release -j8 -p intarsia-backend-hip --test ink_attn"
now() { date '+%F %T'; }
echo "[stage] quick admission $(now)"
bash "$R/run.sh" env INK_ATTN_QUICK=1 $T -- --exact admission_full_sweep --nocapture
echo "[stage] rows-exact $(now)"
bash "$R/run.sh" $T -- --exact rows_exact_real_shapes --nocapture
echo "[stage] bench $(now)"
bash "$R/run.sh" $T -- --exact bench_decode --ignored --nocapture
echo "[stage] full admission $(now)"
bash "$R/run.sh" $T -- --exact admission_full_sweep --nocapture
echo "[stage] done $(now)"
