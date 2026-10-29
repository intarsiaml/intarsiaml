#!/usr/bin/env bash
# Lane rip, a short GPU job (spark2, --quiet gpu,cpu): the kernel-level row-invariance findings and gate (tests/row_invariance.rs),
# then the 8B model-level gate with the ordinary prefill (INTARSIA_RIP_PLAIN=1, the "before" probe) and with row-invariant prefill.
#   OUT=... fleetq run --host spark2 --quiet gpu,cpu --mem 40 --expect 15 --label "rip: findings + model gate" -- scripts/rip/tests.sh
# TESTS="kernel plain model engine decoderows ordinary plan timing" picks the parts (default: kernel plain model).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H=spark2
OUT=${OUT:?}; mkdir -p "$OUT"
M=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes $H "$@"; }
T=${TESTS:-kernel plain model}
F='^test |test result|panicked|FINDING|RIP|matmul_bf16_inv|flash|_bf16:|left|right|row-invariant'
for part in $T; do
  case $part in
    kernel) rsh "$RPATH cd ~/intarsiaml-rip/src && INTARSIA_REQUIRE_GPU=1 INTARSIA_LT_DEBUG=1 cargo test -p intarsia-backend-cuda --release --test row_invariance -- --test-threads=1 --nocapture 2>&1 | grep -aE '$F|matmul_bf16_inv: N'" > "$OUT/kernel.log" ;;
    plain) rsh "$RPATH cd ~/intarsiaml-rip/src && INTARSIA_RIP_PLAIN=1 INTARSIA_QWEN3_DIR=$M INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real real_qwen3_8b_prefill_is_row_invariant -- --test-threads=1 --nocapture 2>&1 | grep -aE '$F'" > "$OUT/plain.log" ;;
    model) rsh "$RPATH cd ~/intarsiaml-rip/src && INTARSIA_QWEN3_DIR=$M INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real real_qwen3_8b_prefill_is_row_invariant -- --test-threads=1 --nocapture 2>&1 | grep -aE '$F'" > "$OUT/model.log" ;;
    engine) rsh "$RPATH cd ~/intarsiaml-rip/src && INTARSIA_QWEN3_DIR=$M INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real -- real_qwen3_8b_reuse_and_busy real_qwen3_8b_batched --test-threads=1 --nocapture 2>&1 | grep -aE '$F|bi8|pairs bitwise'" > "$OUT/engine.log" ;;
    decoderows) rsh "$RPATH cd ~/intarsiaml-rip/src && INTARSIA_QWEN3_DIR=$M INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real real_qwen3_8b_decode_written_rows -- --test-threads=1 --nocapture 2>&1 | grep -aE '$F'" > "$OUT/decoderows.log" ;;
    ordinary) rsh "$RPATH cd ~/intarsiaml-rip/src && INTARSIA_QWEN3_DIR=$M INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real real_qwen3_8b_ordinary_short_pass -- --test-threads=1 --nocapture 2>&1 | grep -aE '$F'" > "$OUT/ordinary.log" ;;
    timing) for rm in ${REFMS:-512 1024 2048 4096}; do rsh "$RPATH cd ~/intarsiaml-rip/src && INTARSIA_INV_REF_M=$rm INTARSIA_LT_DEBUG=1 INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-backend-cuda --release --test row_invariance gemm_inv_timing -- --ignored --nocapture 2>&1 | grep -aE 'TIMING|matmul_bf16_inv: N|test result|panicked'"; done > "$OUT/timing.log" ;;
    plan) rsh "B=~/intarsiaml-rip/bin; for t in base intarsia; do \$B/\$t plan show --backend cuda --model $M --max-seq 8192 --mem-budget 40 < /dev/null > ~/intarsiaml-rip/out/plan8-\$t.txt 2>/dev/null; done; diff ~/intarsiaml-rip/out/plan8-base.txt ~/intarsiaml-rip/out/plan8-intarsia.txt; echo diff-exit \$?" > "$OUT/plan.log" ;;
  esac
  echo "== $part"; tail -4 "$OUT/$part.log"
done
