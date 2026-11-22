#!/usr/bin/env bash
# Lane rocm102, a radiance-down window (STEPS selects; default all). Run by window.sh.
#   smoke   10.2: rocminfo sees gfx1201, `intarsia doctor` (records the ROCm version), `kernels test --quick`. A failure here
#           stops the window (userspace / host-driver mismatch or a broken build: radiance comes back at once)
#   kern    10.2: the full `kernels test --backend hip` (every kernel vs the CPU reference: the direct compiler-regression check)
#   kern7   7.2.4: the same sweep (per-kernel error columns side by side)
#   tests   10.2 (tests7: the same on 7.2.4): the HIP GPU test suite (benches out), tiny-model packed / macro prefill == solo on HIP, the real-35B batched ==
#           local and macro == plain tests (as scripts/gdn-default/stages.sh)
#   full10, full7   the switch qualify (after #486): every stage but kernels (own step) and ab (needs localeval), so acc-long,
#                   acc-4k, envelope, bench and bench-long too. Pass = 10.2 prints the same numbers as 7.2.4 (acc-4k: the same
#                   FAIL as main) and 7.2.4 still hits the checked-in baselines
#   qchk10, qchk7   qualify (tp=2 stage included) against the checked-in 35B HIP baselines, the same tree on each image, so the
#           acc-short / acc-oracle KL and top-1 numbers compare the toolchains directly. Skipped stages: kernels (own step),
#           bench, acc-long, acc-4k (known FAIL on main: restore / aligned), bench-long, ab, envelope
set -u
. "$(dirname "$0")/lib.sh"
cd "$TREE"
STEPS=${STEPS:-"smoke kern tests qchk10 qchk7"}
has() { [[ " $STEPS " == *" $1 "* ]]; }
W=/work/${O#"$TREE"/}
G=/models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
OR=/work/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors
Q="$BIN qualify --backend hip --golden $G --oracle-golden $OR --model-name Qwen3.6-35B-A3B-NVFP4 --mem-budget 29 --bench-sizes 128 --reps 1 --gen 16"
SK="--skip kernels --skip bench --skip acc-long --skip acc-4k --skip bench-long --skip ab --skip envelope"
verdicts() { grep -aE "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" "$O/$1.out" | cut -c1-260 | tee -a "$O/steps.log"; }
log "stages start; $STAMP; STEPS=$STEPS"
if has smoke; then
  run r10 0,1 120 smoke-rocminfo rocminfo; grep -c 'gfx1201' "$O/smoke-rocminfo.out" | sed 's/^/gfx1201 lines: /' | tee -a "$O/steps.log"
  run r10 0 120 smoke-doctor $BIN doctor; grep -aiE 'rocm|hipcc|compiler' "$O/smoke-doctor.out" | head -5 | tee -a "$O/steps.log"
  if ! run r10 0 900 smoke-kernels $BIN kernels test --backend hip --quick; then
    tail -5 "$O/smoke-kernels.out" | tee -a "$O/steps.log"; log "SMOKE FAILED, stopping"; exit 3
  fi
  tail -3 "$O/smoke-kernels.out" | tee -a "$O/steps.log"
fi
if has kern; then run r10 0 1500 kernels $BIN kernels test --backend hip; tail -4 "$O/kernels.out" | tee -a "$O/steps.log"; fi
# kern7: the same sweep on 7.2.4, so the per-kernel max-error columns compare the two compilers case for case
if has kern7; then run r7 0 1500 kernels-r7 $BIN kernels test --backend hip; tail -4 "$O/kernels-r7.out" | tee -a "$O/steps.log"; fi
for ta in r10 r7; do
suf=; [[ $ta == r7 ]] && suf=7
if has "tests$suf"; then
  run $ta 0,1 1500 hip-tests$suf cargo test --release --locked -p intarsia-backend-hip --tests -- --test-threads=1 --skip bench --skip timing
  E="INTARSIA_BATCH_TEST_BACKEND=hip" run $ta 0 600 models-tests$suf cargo test --release --locked --no-fail-fast -p intarsia-models --test batch_recurrent --test mtp -- --test-threads=1 packed_prefill macro_prefill
  E="INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35" run $ta 0 1200 models-tests-35b$suf cargo test --release --locked --no-fail-fast -p intarsia-models --test batch_cuda_real -- --ignored --nocapture --test-threads=1 \
    real_35b_macro_prefill_is_bitwise_the_plain_schedule real_35b_batched_macro_prefill_equals_the_plain_local_engine real_35b_batched_engine_long_prompts_equal_the_local_engine real_35b_batched_decode_is_bitwise_the_solo_step
  grep -ahE "^test result|^test .* \.\.\. FAILED|panicked" "$O"/hip-tests$suf.out "$O"/models-tests*$suf.out | sort | uniq -c | tee -a "$O/steps.log"
fi
done
for a in r10 r7; do
  if has "full${a#r}"; then run $a 0,1 2400 "full-$a" $Q --model $M35 --out "$W/full-$a" --tp 2 --devices 0,1 --skip kernels --skip ab; verdicts "full-$a"; fi
  if has "qchk${a#r}"; then run $a 0,1 1500 "qchk-$a" $Q --model $M35 --out "$W/qchk-$a" --tp 2 --devices 0,1 $SK; verdicts "qchk-$a"; fi
done
log "stages done"
