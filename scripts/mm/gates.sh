#!/bin/bash
# Lane mm gates on endor (inside one exclusive fleetq slot). Usage: scripts/mm/gates.sh OUTDIR [stage ...]
# stages: gb neg kt gc long rec dense ws (default: all). Logs per stage in OUTDIR.
set -u
cd /opt/code/wt-mm
OUT=${1:?outdir}; shift
mkdir -p "$OUT"
ST=${*:-gb neg kt gc long rec dense ws}
export INTARSIA_REQUIRE_GPU=1
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
C="cargo test --release"
for s in $ST; do
  echo "== stage $s $(date +%H:%M:%S)"
  case $s in
    gb) $C -p intarsia-backend-metal --test rows_exact -- --test-threads=1 --nocapture > "$OUT/gb-rows_exact.log" 2>&1; echo "rows_exact exit $?"
        $C -p intarsia-backend-metal --test row_tables -- --test-threads=1 --nocapture > "$OUT/gb-row_tables.log" 2>&1; echo "row_tables exit $?" ;;
    neg) # each negative-control kernel must make its gate FAIL
        for n in fp8_xd moe_gu moe_dn; do
          t=fp8_rows_are_bitwise_the_one_row_gemv; [ "$n" != fp8_xd ] && t=moe_experts_rows_are_bitwise_the_one_token_call
          INTARSIA_METAL_NEGCTL=$n $C -p intarsia-backend-metal --test rows_exact -- --test-threads=1 --exact $t > "$OUT/neg-$n.log" 2>&1
          r=$?; echo "NEGCTL=$n $t exit $r (want non-zero): $(grep -m1 -o 'first differing byte [0-9]* of [0-9]*' "$OUT/neg-$n.log")"
        done ;;
    kt) cargo run --release -p intarsia-cli --features metal -- kernels test --backend metal > "$OUT/kernels-test.log" 2>&1; echo "kernels test exit $?"; tail -3 "$OUT/kernels-test.log" ;;
    gc) INTARSIA_QWEN36_DIR=$M35 $C -p intarsia-models --test batch_metal_real -- --test-threads=1 --nocapture > "$OUT/gc-35b.log" 2>&1; echo "G-C exit $?"; grep -E "test result|bitwise identical|identical" "$OUT/gc-35b.log" | tail -5 ;;
    long) INTARSIA_QWEN36_DIR=$M35 $C -p intarsia-models --test batch_metal_real -- --ignored --nocapture --test-threads=1 --exact real_35b_batched_engine_long_prompts_equal_the_local_engine > "$OUT/gc-long.log" 2>&1; echo "long exit $?"; grep -E "test result" "$OUT/gc-long.log" ;;
    rec) INTARSIA_BATCH_TEST_BACKEND=metal $C -p intarsia-models --test batch_recurrent -- --test-threads=1 > "$OUT/batch_recurrent.log" 2>&1; echo "batch_recurrent exit $?"; grep "test result" "$OUT/batch_recurrent.log" ;;
    dense) INTARSIA_TEST_REAL_MODEL=1 $C -p intarsia-models --test batch_metal -- --test-threads=1 --nocapture > "$OUT/dense-8b.log" 2>&1; echo "dense exit $?"; grep -E "test result|identical" "$OUT/dense-8b.log" | tail -3 ;;
    ws) $C --workspace --no-fail-fast > "$OUT/workspace.log" 2>&1; echo "workspace exit $?"; grep -E "^test result" "$OUT/workspace.log" | awk '{p+=$4; f+=$6; i+=$8} END {print "passed", p, "failed", f, "ignored", i}' ;;
  esac
done
echo "== done $(date +%H:%M:%S)"
