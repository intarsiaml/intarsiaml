#!/usr/bin/env bash
# Lane l5r7: the batched == solo gates alone (G-C), after wait_vram. Also the plain macro-prefill identity test of main (ignored test), as context.
. "$HOME/intarsiaml-l5-$NAME/scripts/l5r7/lib.sh"
wait_vram 60
T="env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35"
echo "[gc] $(now) G-C: 35B batched decode == solo step + batched engine == local engine"
dv $T cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture 2>&1 | tee "$O/batch_cuda_real.log" | grep -aE "^test |^test result|FAILED|panicked|^error|bitwise identical|pairs|Memory"
echo "[gc] done $(now)"
