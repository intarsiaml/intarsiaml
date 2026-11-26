#!/usr/bin/env bash
# Window 2 stage (lane ink-hip-dec2): bitwise tests of the three forms on the R9700 (ink_moe rows-exact + odd ids, all of ink_mix incl.
# the moe_route fast tail and the blend tails), the experts bench at T 1 / 2 / 4 with the forms off and on, then a decode A/B
# in the branch binary by INTARSIA_HIP_INK_DEC_FAST (0 = main's forms, 7 = all, 6 / 5 / 3 = one bit off). Radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-dec2
S=$REPO/.scratch
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-dec2a
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0"
now() { date '+%F %T'; }
spd() { grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)|steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+|Error.*|panicked.*' | tr '\n' ' '; echo; }
echo "[stage] ink_moe tests $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_moe -- --nocapture --test-threads=1 ink_moe_rows_exact ink_moe_wave_grouping_odd_ids 2>&1 | grep -aE "test |test result|bitwise|panicked|assert|error" | head -20
echo "[stage] ink_mix tests $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_mix -- --nocapture --test-threads=1 2>&1 | grep -aE "^test |test result|bitwise|panicked|assert|ADMIT|FAIL|error" | head -30
for f in 0 7; do
  echo "[stage] ink_moe_bench T=1,2,4 INTARSIA_HIP_INK_DEC_FAST=$f $(now)"
  bash "$S/run.sh" env INTARSIA_HIP_INK_DEC_FAST=$f INK_MOE_BENCH=1 INK_MOE_BENCH_T=1,2,4 INK_MOE_BENCH_NOSWEEP=1 cargo test --release -p intarsia-backend-hip --test ink_moe -- --nocapture --test-threads=1 ink_moe_bench 2>&1 | grep -aE "INKMOEBENCH|panicked" | head -16
done
for rep in 1 2; do
  for f in 0 7 6 5 3; do
    echo "[stage] short f=$f rep $rep $(now) $(bash "$S/run.sh" env INTARSIA_HIP_INK_DEC_FAST=$f $B $G --max-tokens 128 -p "$PS" 2>&1 | spd)"
  done
  for f in 0 7; do
    echo "[stage] 4k f=$f rep $rep $(now) $(bash "$S/run.sh" env INTARSIA_HIP_INK_DEC_FAST=$f $B $G --max-tokens 128 -p "$P4" 2>&1 | spd)"
  done
done
echo "[stage] done $(now)"
