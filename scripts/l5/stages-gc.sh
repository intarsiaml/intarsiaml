#!/usr/bin/env bash
# Lane l5 gate window A (knob ON): the device test of the kernel through the registry, the 35B batched-decode == solo-step test (G-C, 17,550 pairs),
# and MTP ids equal to the plain ids within the arm (3.5k prompt: the prefill chunks take the fp8 experts, decode and the verify rows the bf16 rows).
# INTARSIA_HIP_MOE_FP8_MIN=16 for the tests whose prompts are short; the 4k ids run at the default threshold (1025).
. "$HOME/intarsiaml-l5-$NAME/scripts/l5/lib.sh"
PD=$HOME/intarsiaml-l5-$NAME/scripts/hg/prompts
ON="INTARSIA_HIP_MOE_FP8_WMMA=1"
echo "[gc] $(now) moe_f8 device test"
dv env $ON cargo test --release -p intarsia-backend-hip --test moe_f8 -- --test-threads=1 --nocapture 2>&1 | grep -aE "^test |^test result|FAILED|panicked|rel_l2|^error" | tee "$O/moe_f8.log"
echo "[gc] $(now) G-C: 35B batched decode == solo step, knob on (MIN=16)"
T="env $ON INTARSIA_HIP_MOE_FP8_MIN=16 INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35"
dv $T cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture 2>&1 | tee "$O/batch_cuda_real.log" | grep -aE "^test |^test result|FAILED|panicked|^error|bitwise identical|pairs"
echo "[gc] $(now) G-C ignored long-prompt test"
dv $T cargo test --release -p intarsia-models --test batch_cuda_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine --test-threads=1 --nocapture 2>&1 | tee "$O/batch_cuda_real_long.log" | grep -aE "^test |^test result|FAILED|panicked|^error"
echo "[gc] $(now) MTP ids vs plain ids, knob on, 35B, 3.5k prompt"
gen dv n35-4k-on-plain "$M35" "$PD/4k.txt" 256 --mem-budget 29
gen dv n35-4k-on-mtp "$M35" "$PD/4k.txt" 256 --mem-budget 29 --spec mtp --spec-k 2
cmp -s "$O/gen-n35-4k-on-plain.ids" "$O/gen-n35-4k-on-mtp.ids" && echo "MTP ids == plain ids (knob on): IDENTICAL" || echo "MTP ids vs plain ids (knob on): DIFFER"
grep -a -E '^spec:' "$O/gen-n35-4k-on-mtp.err" | tail -1 | cut -c1-200
echo "[gc] done $(now)"
