#!/usr/bin/env bash
# Window 2 stage (lane ink-hip-dec3): decode A/B in the branch binary by INTARSIA_HIP_INK_DEC3 (0 = main's forms, 3 = all, 1 / 2 = one
# bit), 128 tokens, 2 processes per setting, interleaved; then the bitwise tests on the R9700 (ink_mix: rows-exact with the knob at 0,
# moe_route with the knob at 0, argmax every mask; ink_attn rows_exact_real_shapes: op 75 with DPP vs the unfused pair, and at 0).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-dec3
S=$REPO/.scratch
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-${BR:-dec3a}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0"
now() { date '+%F %T'; }
spd() { grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)|steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+|Error.*|panicked.*' | tr '\n' ' '; echo; }
for rep in 1 2; do
  for f in 0 3 1 2; do
    echo "[stage] short f=$f rep $rep $(now) $(bash "$S/run.sh" env INTARSIA_HIP_INK_DEC3=$f $B $G --max-tokens 128 -p "$PS" 2>&1 | spd)"
  done
  for f in 0 3; do
    echo "[stage] 4k f=$f rep $rep $(now) $(bash "$S/run.sh" env INTARSIA_HIP_INK_DEC3=$f $B $G --max-tokens 128 -p "$P4" 2>&1 | spd)"
  done
done
echo "[stage] ink_mix tests $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_mix -- --nocapture --test-threads=1 --skip bench 2>&1 | grep -aE "^test |test result|bitwise|panicked|assert|ADMIT|FAIL|error" | head -30
echo "[stage] ink_attn rows_exact_real_shapes $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_attn -- --nocapture --test-threads=1 rows_exact_real_shapes 2>&1 | grep -aE "^test |test result|bitwise|panicked|assert|error" | head -12
echo "[stage] done $(now)"
