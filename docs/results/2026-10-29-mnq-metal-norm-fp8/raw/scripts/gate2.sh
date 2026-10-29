#!/bin/bash
# gates 2: metal tests + real-model batch/reuse gates + kernels test + plan show diff. Run in a fleetq slot.
[ -n "${FLEETQ_JOB:-}" ] || { echo "not in slot"; exit 2; }
cd /opt/code/intarsiaml/.worktrees/mnq
export CARGO_TARGET_DIR=$PWD/.lane/target
O=.lane/g2
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
t() { echo "## $*"; "$@" 2>&1 | tee -a $O/all.log | grep -E "^test |test result|panicked|FAILED" ; }
for tf in rows_exact row_tables token_feedback norm_fp8 fused dense lowlat qkv_fold multirow gdn; do
  echo "## metal $tf (INTARSIA_METAL_NORM_FP8=1)"; INTARSIA_METAL_NORM_FP8=1 cargo test --release -p intarsia-backend-metal --test $tf 2>&1 | tee $O/metal-$tf.log | grep -E "test result|FAILED|panicked"
done
echo "## metal norm_fp8 level 2"; INTARSIA_METAL_NORM_FP8=2 cargo test --release -p intarsia-backend-metal --test norm_fp8 2>&1 | tee $O/metal-norm_fp8-l2.log | grep -E "test result|FAILED|panicked"
echo "## rows_exact level 0"; INTARSIA_METAL_NORM_FP8=0 cargo test --release -p intarsia-backend-metal --test rows_exact 2>&1 | tee $O/metal-rows_exact-l0.log | grep -E "test result|FAILED"
for tf in batch_metal_real batch_recurrent exact_reuse; do
  echo "## models $tf"; INTARSIA_METAL_NORM_FP8=1 INTARSIA_TEST_REAL_MODEL=1 cargo test --release -p intarsia-models --test $tf 2>&1 | tee $O/models-$tf.log | grep -E "^test |test result|FAILED|panicked|SKIPPED"
done
echo "## kernels test"
.lane/bin/intarsia-dev kernels test --backend metal --quick > $O/kernels-dev.log 2>&1; echo "exit $?"; tail -3 $O/kernels-dev.log | cut -c1-200
echo "## plan show"
for l in 1 0; do INTARSIA_METAL_NORM_FP8=$l .lane/bin/intarsia-dev plan show --model $M --backend metal > $O/plan-dev-$l.txt 2>/dev/null; done
.lane/bin/intarsia-main plan show --model $M --backend metal > $O/plan-main.txt 2>/dev/null
diff $O/plan-main.txt $O/plan-dev-1.txt > $O/plan-diff-1.txt; echo "plan diff default: $(wc -l < $O/plan-diff-1.txt) lines"
diff $O/plan-main.txt $O/plan-dev-0.txt > $O/plan-diff-0.txt; echo "plan diff =0: $(wc -l < $O/plan-diff-0.txt) lines"
echo GATE2-DONE
