#!/usr/bin/env bash
# Lane a4k-restore, the radiance-down window (STEPS selects; default all):
#   a4k     the acc-4k (prefix-reuse) qualify stage with this build (macro prefill on, the HIP recipe default): restore / aligned
#           cases must now reuse (chunk 4096: chat_reuse_4k; the stage's chunk-1024 run: long_docs_4k, long_code_4k)
#   a4k0    the same with INTARSIA_PREFILL_MACRO=0 (control: the plain chunk schedule, which always kept chunk-end snapshots)
#   tests   the HIP GPU test suite (benches left out); the tiny-model macro / packed prefill tests (batch_recurrent in full,
#           incl. the new macro_prefill_keeps_chunk_snapshots_for_reuse; mtp's packed / macro ones) on HIP; the real-35B macro ==
#           plain schedule and batched == local engine tests; the real-35B hybrid prefix-reuse gate (engine_hybrid_reuse_equals_cold)
#   mytest  batch_recurrent's macro_prefill_keeps_chunk_snapshots_for_reuse alone on HIP
#   hrbase  engine_hybrid_reuse_equals_cold on HIP from the origin/main tree ~/intarsiaml-a4k-restore-base (build-base.sh), and on
#           this tree with INTARSIA_PREFILL_MACRO=0: is its HIP failure this branch's?
set -u
. "$(dirname "$0")/lib.sh"
cd "$TREE"
STEPS=${STEPS:-"a4k a4k0 tests"}
has() { [[ " $STEPS " == *" $1 "* ]]; }
W=/work/${O#"$TREE"/}
G=/models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
OR=/work/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors
Q="$BIN qualify --backend hip --golden $G --oracle-golden $OR --model-name Qwen3.6-35B-A3B-NVFP4 --mem-budget 29 --bench-sizes 128 --reps 1 --gen 16"
SKA="--skip kernels --skip gate --skip acc-short --skip acc-oracle --skip bench --skip acc-long --skip envelope --skip bench-long --skip ab --skip tp"
log "stages start; $STAMP; STEPS=$STEPS"
if has a4k; then
  run 0 1200 q-a4k $Q --model $M35 --out $W/q-a4k $SKA
  grep -aE "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" "$O/q-a4k.out" | cut -c1-260 | tee -a "$O/steps.log"
fi
if has a4k0; then
  E="INTARSIA_PREFILL_MACRO=0" run 0 1200 q-a4k-macro0 $Q --model $M35 --out $W/q-a4k-macro0 $SKA
  grep -aE "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" "$O/q-a4k-macro0.out" | cut -c1-260 | tee -a "$O/steps.log"
fi
if has tests; then
  NB="--skip bench --skip timing"
  run 0,1 1500 hip-tests cargo test --release -p intarsia-backend-hip --tests -- --test-threads=1 $NB
  E="INTARSIA_BATCH_TEST_BACKEND=hip" run 0 900 models-tests cargo test --release --no-fail-fast -p intarsia-models --test batch_recurrent -- --test-threads=1
  E="INTARSIA_BATCH_TEST_BACKEND=hip" run 0 600 models-tests-mtp cargo test --release --no-fail-fast -p intarsia-models --test mtp -- --test-threads=1 packed_prefill macro_prefill
  T35="INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35"
  E="$T35" run 0 2400 models-tests-35b cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real -- --ignored --nocapture --test-threads=1 \
    real_35b_macro_prefill_is_bitwise_the_plain_schedule real_35b_batched_macro_prefill_equals_the_plain_local_engine real_35b_batched_engine_long_prompts_equal_the_local_engine
  E="INTARSIA_PROBE_BACKEND=hip INTARSIA_PROBE_DIR=$M35 INTARSIA_REQUIRE_GPU=1" run 0 900 hybrid-reuse cargo test --release --no-fail-fast -p intarsia-models --test hybrid_reuse_real -- --nocapture --test-threads=1 engine_hybrid_reuse_equals_cold
  grep -ahE "^test result|FAILED|panicked|^test .* \.\.\. FAILED|SKIPPED" "$O"/hip-tests.out "$O"/models-tests*.out "$O"/hybrid-reuse.out "$O"/hybrid-reuse.err | sort | uniq -c | tee -a "$O/steps.log"
fi
if has mytest; then
  E="INTARSIA_BATCH_TEST_BACKEND=hip" run 0 600 models-mytest cargo test --release --no-fail-fast -p intarsia-models --test batch_recurrent -- --test-threads=1 macro_prefill_keeps
  grep -ahE "^test |^test result" "$O/models-mytest.out" | tee -a "$O/steps.log"
fi
if has hrbase; then
  HR="INTARSIA_PROBE_BACKEND=hip INTARSIA_PROBE_DIR=$M35 INTARSIA_REQUIRE_GPU=1"
  INTARSIA_REPO=$HOME/intarsiaml-a4k-restore-base INTARSIA_TARGET_VOL=intarsia-cargo-target-a4k-restore-base E="$HR" run 0 900 hybrid-reuse-base \
    cargo test --release --no-fail-fast -p intarsia-models --test hybrid_reuse_real -- --nocapture --test-threads=1 engine_hybrid_reuse_equals_cold
  E="$HR INTARSIA_PREFILL_MACRO=0" run 0 900 hybrid-reuse-macro0 cargo test --release --no-fail-fast -p intarsia-models --test hybrid_reuse_real -- --nocapture --test-threads=1 engine_hybrid_reuse_equals_cold
  grep -ahE "^test |^test result|panicked|resumed" "$O"/hybrid-reuse-base.* "$O"/hybrid-reuse-macro0.* | cut -c1-200 | tee -a "$O/steps.log"
fi
log "stages done"
