#!/usr/bin/env bash
# Lane gdn-default, the radiance-down window (STEPS selects; default all):
#   tests   the HIP GPU test suite (benches left out); the tiny-model packed / macro / MTP prefill == solo engine tests on HIP,
#           also with INTARSIA_GDN_CHUNK_MIN=1 so their short segments take the chunked kernel; the real-35B batched == local
#           engine and macro-prefill == plain-schedule tests (scripts/l7m/gates.sh), the macro bitwise one also with the knob off
#   kern    `kernels test --backend hip --quick` (the default selection now includes the chunked delta rule)
#   rec     qualify (tp=2 stage included) with the four 35B HIP baselines moved away -> MISSING-BASELINE; it records the
#           acc-short baseline + envelope and the tp=1 / tp=2 CPU-oracle dumps (as scripts/fx1-oracle/hip-window2.sh did)
#   chk     the recorded files installed in this tree's golden/baselines/, qualify again -> PASS expected
#   a4k     the acc-4k (prefix-reuse) stage, which the gdn-chunk lane never ran
set -u
. "$(dirname "$0")/lib.sh"
cd "$TREE"
STEPS=${STEPS:-"tests kern rec chk a4k"}
has() { [[ " $STEPS " == *" $1 "* ]]; }
W=/work/${O#"$TREE"/}
G=/models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors   # sha256 b6b51e1b..., the golden the old baseline used
OR=/work/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors
Q="$BIN qualify --backend hip --golden $G --oracle-golden $OR --model-name Qwen3.6-35B-A3B-NVFP4 --mem-budget 29 --bench-sizes 128 --reps 1 --gen 16"
SK="--skip kernels --skip bench --skip acc-long --skip acc-4k --skip bench-long --skip ab"
B35=(qwen36-35b-a3b-nvfp4-vllm-top20.hip.baseline.json qwen36-35b-a3b-nvfp4-vllm-top20.hip.envelope.json
     qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors qwen36-35b-a3b-nvfp4-cpu-oracle.hip-tp2.selfgolden.safetensors)
log "stages start; $STAMP; STEPS=$STEPS"
if has tests; then
  NB="--skip bench --skip timing"
  run 0,1 1500 hip-tests cargo test --release -p intarsia-backend-hip --tests -- --test-threads=1 $NB
  for e in "" "INTARSIA_GDN_CHUNK_MIN=1"; do
    E="INTARSIA_BATCH_TEST_BACKEND=hip $e" run 0 600 "models-tests${e:+-min1}" cargo test --release --no-fail-fast -p intarsia-models --test batch_recurrent --test mtp -- --test-threads=1 packed_prefill macro_prefill
  done
  T35="INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35"
  E="$T35" run 0 2400 models-tests-35b cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real -- --ignored --nocapture --test-threads=1 \
    real_35b_macro_prefill_is_bitwise_the_plain_schedule real_35b_batched_macro_prefill_equals_the_plain_local_engine real_35b_batched_engine_long_prompts_equal_the_local_engine real_35b_batched_decode_is_bitwise_the_solo_step
  E="$T35 INTARSIA_GDN_CHUNK=0" run 0 900 models-tests-35b-off cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real -- --ignored --nocapture --test-threads=1 real_35b_macro_prefill_is_bitwise_the_plain_schedule
  grep -ahE "^test result|FAILED|panicked|^test .* \.\.\. FAILED" "$O"/hip-tests.out "$O"/models-tests*.out | sort | uniq -c | tee -a "$O/steps.log"
fi
if has kern; then run 0 900 kernels $BIN kernels test --backend hip --quick; tail -3 "$O/kernels.out" | tee -a "$O/steps.log"; fi
if has rec; then
  mkdir -p "$O/old-baselines"; for f in "${B35[@]}"; do mv "golden/baselines/$f" "$O/old-baselines/" 2>/dev/null; done
  run 0,1 1800 q-rec $Q --model $M35 --out $W/q-rec --tp 2 --devices 0,1 $SK
  grep -aE "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" "$O/q-rec.out" | cut -c1-260 | tee -a "$O/steps.log"
  Bd=$O/q-rec/baselines
  cp "$Bd/acc-short.baseline.json" "$O/${B35[0]}"; cp "$Bd/envelope.json" "$O/${B35[1]}"; cp "$Bd/${B35[2]}" "$Bd/${B35[3]}" "$O/"
  log "recorded: $(cd "$O" && ls -la "${B35[@]}" 2>&1 | awk '{print $5, $NF}' | tr '\n' ' ')"
fi
if has chk; then
  for f in "${B35[@]}"; do cp "$O/$f" golden/baselines/; done
  run 0,1 1500 q-chk $Q --model $M35 --out $W/q-chk --tp 2 --devices 0,1 $SK --skip envelope
  grep -aE "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" "$O/q-chk.out" | cut -c1-260 | tee -a "$O/steps.log"
fi
if has a4k; then
  run 0 1200 q-a4k $Q --model $M35 --out $W/q-a4k --skip kernels --skip gate --skip acc-short --skip acc-oracle --skip bench --skip acc-long --skip envelope --skip bench-long --skip ab --skip tp
  grep -aE "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" "$O/q-a4k.out" | cut -c1-260 | tee -a "$O/steps.log"
fi
log "stages done"
