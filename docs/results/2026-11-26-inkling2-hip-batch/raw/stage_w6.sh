#!/usr/bin/env bash
# Window 6 stage (lane ink-hip-batch): b6 = the row-table launchers in their own translation units (ink_mix_rows.hip, ink_attn_rows.hip),
# so ink_mix.hip / ink_attn.hip compile to main's device code, kernel for kernel (.scratch/isa3.sh). The unit gate of the row-table
# kernels (the RT instantiations now in the new modules), then the solo gate against main (stage_gate.sh: greedy md5 64 / 256, short / 4k,
# decode tok/s ABBA 4 + 4).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-batch
S=$REPO/.scratch
BR=${BR:-b6}
now() { date '+%F %T'; }
f() { grep -aE "^test |test result|bitwise|row-table calls|panicked|assert|SKIPPED|error(\[|:)" | head -${1:-30}; }
echo "[stage] unit: ink_rows (row-table TUs) $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_rows -- --nocapture --test-threads=1 2>&1 | f 12
echo "[stage] unit: ink_attn rows_exact_real_shapes $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_attn -- --nocapture --test-threads=1 rows_exact_real_shapes 2>&1 | f 6
for op in conv_mix ink_attn_decode; do
  echo "[stage] kernels test $op $(now)"
  bash "$S/run.sh" /work/.scratch/bin/intarsia-$BR kernels test --backend hip --op $op 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked" | head -4
done
echo "[stage] inkling2_batch tiny on hip $(now)"
bash "$S/run.sh" env INTARSIA_INKLING2_BACKEND=hip CI=1 cargo test --release -j8 -p intarsia-models --test inkling2_batch -- --nocapture --test-threads=1 2>&1 | f 14
echo "[stage] solo gate $(now)"
BR=$BR MA=${MA:-main2} bash "$S/stage_gate.sh"
