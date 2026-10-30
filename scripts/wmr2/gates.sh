#!/usr/bin/env bash
# Lane wmr, ON tatooine (copied to ~/wmr/ by guarded.sh) inside a window with radiance down. Groups (comma separated, run in parallel where noted):
#   R  card 1: rows_exact + wmma_folds + frag_layout (+ quick_coverage_wmma with the knob on), knob unset / ROWS=1 alone / option 1
#   B  card 0: 35B batch_cuda_real "batched" (batched == solo, bitwise, one process) knob unset and option 1
#   P  card 0: `plan show` (text + json) knob unset / ROWS=1 alone / option 1, --max-batch 16
# Logs: ~/wmr-logs/gates/. "Option 1" = INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all.
set -uo pipefail
G=${1:-R,B,P}
O=$HOME/wmr-logs/gates; mkdir -p "$O"
TREE=$HOME/intarsiaml-tp2-wmr; DEVSH=$TREE/deploy/tatooine/dev.sh
REL=intarsia-cargo-target-tp2-wmr; TST=intarsia-cargo-target-tp2-wmr-hiptest
M35=/models/Qwen3.6-35B-A3B-NVFP4
OPT1="INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all"
BARE="INTARSIA_HIP_WMMA_ROWS=1"
now() { date '+%F %T'; }
dev() { local g=$1 vol=$2; shift 2; INTARSIA_CONTAINER_NAME=wmr-job-$g INTARSIA_REPO=$TREE INTARSIA_TARGET_VOL=$vol HIP_VISIBLE_DEVICES=$g $DEVSH "$@" < /dev/null; }
pids=()
if [[ $G == *R* ]]; then
  (
    for arm in off bare opt1; do
      case $arm in off) e="NOKNOB=1";; bare) e=$BARE;; opt1) e=$OPT1;; esac
      echo "== R $arm ($e) $(now)"
      dev 1 $TST env $e INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_DEVICE=0 cargo test --release --locked -p intarsia-backend-hip --test rows_exact --test wmma_folds --test frag_layout --no-fail-fast -- --test-threads=1 > "$O/R-$arm.log" 2>&1
      echo "exit=$? $(grep -aE '^test result' "$O/R-$arm.log" | tr '\n' ' ')"
      grep -aE "^test .* (ok|FAILED|ignored)$|panicked" "$O/R-$arm.log" | head -80
    done
    echo "== R quick_coverage (knob unset: skips *wmma*) and quick_coverage_wmma (option 1) $(now)"
    dev 1 $TST env NOKNOB=1 INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_DEVICE=0 cargo test --release --locked -p intarsia-backend-hip --test quick_coverage --no-fail-fast -- --test-threads=1 > "$O/R-qc-off.log" 2>&1
    echo "exit=$? $(grep -aE '^test result' "$O/R-qc-off.log" | tr '\n' ' ')"
    dev 1 $TST env $OPT1 INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_DEVICE=0 cargo test --release --locked -p intarsia-backend-hip --test quick_coverage_wmma --no-fail-fast -- --test-threads=1 > "$O/R-qc-wmma.log" 2>&1
    echo "exit=$? $(grep -aE '^test result' "$O/R-qc-wmma.log" | tr '\n' ' ')"
    echo "R done $(now)"
  ) > "$O/R.out" 2>&1 &
  pids+=($!)
fi
if [[ $G == *B* ]]; then
  (
    T="env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35"
    for arm in opt1 off; do
      case $arm in off) e="NOKNOB=1";; opt1) e=$OPT1;; esac
      echo "== B batch_cuda_real batched, $arm ($e) $(now)"
      dev 0 $TST env $e INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 cargo test --release --locked -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture batched > "$O/B-$arm.log" 2>&1
      echo "exit=$? $(grep -aE '^test result' "$O/B-$arm.log" | tr '\n' ' ')"
      grep -aE "^test .* (ok|FAILED|ignored)$|panicked" "$O/B-$arm.log" | head -40
    done
    echo "B done $(now)"
  ) > "$O/B.out" 2>&1 &
  pids+=($!)
fi
if [[ $G == *P* ]]; then
  for arm in off bare opt1; do
    case $arm in off) e="NOKNOB=1";; bare) e=$BARE;; opt1) e=$OPT1;; esac
    dev 0 $REL env $e /target/release/intarsia plan show --backend hip --model $M35 --max-seq 8192 --max-batch 16 > "$O/plan-$arm.txt" 2> "$O/plan-$arm.err" ||
      dev 0 $REL env $e /target/release/intarsia plan show --backend hip --model $M35 --max-seq 8192 > "$O/plan-$arm.txt" 2> "$O/plan-$arm.err"
    echo "plan show $arm exit=$? $(wc -c < "$O/plan-$arm.txt") B"
  done
  for arm in bare opt1; do cmp -s "$O/plan-off.txt" "$O/plan-$arm.txt" && echo "plan show: $arm IDENTICAL to knob unset" || echo "plan show: $arm DIFFERS from knob unset"; done
fi
for p in "${pids[@]}"; do wait "$p"; done
for f in R B; do [[ -s $O/$f.out ]] && { echo "######## $f"; cat "$O/$f.out"; }; done
echo "gates done $(now)"
