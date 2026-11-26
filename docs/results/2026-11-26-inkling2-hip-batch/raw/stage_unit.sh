#!/usr/bin/env bash
# Window 1 stage (lane ink-hip-batch): the row-table kernels' bitwise tests and the regressions of the solo kernels they are
# instantiated from, admission of the two base ops, the tiny-fixture batched == solo tests on HIP. Radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-batch
S=$REPO/.scratch
BR=${BR:-b2}
now() { date '+%F %T'; }
T="env INTARSIA_REQUIRE_GPU=1 cargo test --release -j8"
f() { grep -aE "^test |test result|bitwise|row-table calls|panicked|assert|ADMIT|REJECT|FAIL|SKIPPED|error(\[|:)" | head -${1:-30}; }
echo "[stage] ink_rows $(now)"
bash "$S/run.sh" $T -p intarsia-backend-hip --test ink_rows -- --nocapture --test-threads=1 2>&1 | f 20
echo "[stage] ink_attn rows_exact_real_shapes $(now)"
bash "$S/run.sh" $T -p intarsia-backend-hip --test ink_attn -- --nocapture --test-threads=1 rows_exact_real_shapes 2>&1 | f 12
echo "[stage] ink_mix (no bench) $(now)"
bash "$S/run.sh" $T -p intarsia-backend-hip --test ink_mix -- --nocapture --test-threads=1 --skip bench 2>&1 | f 30
for op in conv_mix ink_attn_decode; do
  echo "[stage] kernels test $op $(now)"
  bash "$S/run.sh" /work/.scratch/bin/intarsia-$BR kernels test --backend hip --op $op 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked" | head -6
done
echo "[stage] inkling2_batch tiny on hip $(now)"
bash "$S/run.sh" env INTARSIA_INKLING2_BACKEND=hip CI=1 cargo test --release -j8 -p intarsia-models --test inkling2_batch -- --nocapture --test-threads=1 2>&1 | f 30
echo "[stage] done $(now)"
