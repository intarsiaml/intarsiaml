#!/usr/bin/env bash
# Lane l5r7: the bitwise launch-order test on the device (tests/moe_pairs.rs: the z-fast order against the old order for every kernel family, both tiles,
# with and without pieces), the registry accuracy test (tests/quantmoe.rs), and the 35B batched == solo gates (G-C: decode steps, 17,550 pairs; the ignored
# long-prompt engine test) on the branch build with the default launch order.
. "$HOME/intarsiaml-l5-$NAME/scripts/l5r7/lib.sh"
echo "[gc] $(now) moe_pairs + quantmoe"
dv cargo test --release -p intarsia-backend-hip --test moe_pairs --test quantmoe -- --test-threads=1 --nocapture 2>&1 | tee "$O/hip_tests.log" | grep -aE "^test |^test result|FAILED|panicked|^error"
echo "[gc] $(now) G-C: 35B batched decode == solo step"
T="env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35"
dv $T cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture 2>&1 | tee "$O/batch_cuda_real.log" | grep -aE "^test |^test result|FAILED|panicked|^error|bitwise identical|pairs"
echo "[gc] $(now) G-C ignored long-prompt engine test"
dv $T cargo test --release -p intarsia-models --test batch_cuda_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine --test-threads=1 --nocapture 2>&1 | tee "$O/batch_cuda_real_long.log" | grep -aE "^test |^test result|FAILED|panicked|^error"
echo "[gc] done $(now)"
