#!/bin/bash
# G3: batched rows equal solo etc. on the candidate (inside a fleetq slot). usage: gates.sh OUTDIR stage...
cd /opt/code/intarsiaml/.worktrees/mgd
export CARGO_TARGET_DIR=$PWD/target INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1
O=$1; shift
for st in "$@"; do
  echo "== stage $st $(date +%T)"
  case $st in
    gb) cargo test --release -p intarsia-backend-metal --test gdn --test rows_exact --test row_tables > $O/gb.log 2>&1; echo "gb exit $?"; grep -E "^test result|FAILED" $O/gb.log ;;
    gc) cargo test --release -p intarsia-models --test batch_metal_real -- --test-threads=1 > $O/gc1.log 2>&1; echo "gc exit $?"; grep -E "^test result|FAILED|bitwise" $O/gc1.log | cut -c1-200 ;;
    long) cargo test --release -p intarsia-models --test batch_metal_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine > $O/long.log 2>&1; echo "long exit $?"; grep -E "^test result|FAILED" $O/long.log ;;
    rec) cargo test --release -p intarsia-models --test batch_recurrent > $O/rec.log 2>&1; echo "rec exit $?"; grep -E "^test result|FAILED" $O/rec.log ;;
    dense) cargo test --release -p intarsia-models --test batch_metal > $O/dense.log 2>&1; echo "dense exit $?"; grep -E "^test result|FAILED" $O/dense.log ;;
  esac
done
echo "== done $(date +%T)"
