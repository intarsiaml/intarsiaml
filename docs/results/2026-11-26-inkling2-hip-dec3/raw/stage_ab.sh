#!/usr/bin/env bash
# Window 3 stage (lane ink-hip-dec3): on the dec3b binary (DPP butterflies + argmax loads + the unembed in pieces): the bitwise tests
# (all of ink_mix incl. gemv_split_bitwise / argmax_dec3_bitwise / moe_route / rows-exact at INTARSIA_HIP_INK_DEC3=0; ink_attn
# rows_exact_real_shapes), a rocprofv3 trace of the short decode, then decode A/B: main's binary and INTARSIA_HIP_INK_DEC3 = 0, 7, 3 (no split)
# in the branch, 128 tokens, 2 processes per setting, interleaved. Radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-dec3
S=$REPO/.scratch
OUT=$REPO/logs/w3
rm -rf "$OUT"; mkdir -p "$OUT"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-${BR:-dec3b}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0"
now() { date '+%F %T'; }
spd() { grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)|steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+|Error.*|panicked.*' | tr '\n' ' '; echo; }
echo "[stage] ink_mix tests $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_mix -- --nocapture --test-threads=1 --skip bench 2>&1 | grep -aE "^test |test result|bitwise|panicked|assert|FAIL|error\[" | head -30
echo "[stage] ink_attn rows_exact_real_shapes $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_attn -- --nocapture --test-threads=1 rows_exact_real_shapes 2>&1 | grep -aE "^test |test result|bitwise|panicked|assert|error\[" | head -12
echo "[stage] rows_exact: bf16 matmul / logits rows vs the one-row GEMV (now in pieces at the big heads), row-parallel ops (argmax) $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test rows_exact -- --nocapture --test-threads=1 bf16_matmul_and_logits_rows_are_bitwise_the_one_row_gemv row_parallel_ops_are_row_independent 2>&1 | grep -aE "^test |test result|bitwise|panicked|assert|error\[" | head -12
echo "[stage] rocprof short $(now)"
mkdir -p "$OUT/rp-short"
INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 bash "$S/run.sh" rocprofv3 --kernel-trace --output-format csv -d /work/logs/w3/rp-short -o run -- $B $G --max-tokens 24 -p "$PS" 2>&1 | grep -aE "finish=|steps=|Error|panicked" | cut -c1-300
csv=$(find "$OUT/rp-short" -name '*kernel_trace.csv' | head -1)
[ -n "$csv" ] && gzip -kf "$csv" && python3 "$S/ktrace.py" "$csv" 8 | head -14
BM=/work/.scratch/bin/intarsia-main
for rep in 1 2; do
  echo "[stage] short main rep $rep $(now) $(bash "$S/run.sh" $BM $G --max-tokens 128 -p "$PS" 2>&1 | spd)"
  for f in 0 7 3; do
    echo "[stage] short f=$f rep $rep $(now) $(bash "$S/run.sh" env INTARSIA_HIP_INK_DEC3=$f $B $G --max-tokens 128 -p "$PS" 2>&1 | spd)"
  done
  echo "[stage] 4k main rep $rep $(now) $(bash "$S/run.sh" $BM $G --max-tokens 128 -p "$P4" 2>&1 | spd)"
  for f in 0 7; do
    echo "[stage] 4k f=$f rep $rep $(now) $(bash "$S/run.sh" env INTARSIA_HIP_INK_DEC3=$f $B $G --max-tokens 128 -p "$P4" 2>&1 | spd)"
  done
done
echo "[stage] done $(now)"
