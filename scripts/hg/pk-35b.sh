#!/usr/bin/env bash
# Lane hg: the 35B packed-prefill engine test, full output (radiance must already be down, inside the fleetq job).
set -uo pipefail
INTARSIA_REPO=$HOME/intarsiaml-hgpk INTARSIA_TARGET_VOL=intarsia-cargo-target-hgpk HIP_VISIBLE_DEVICES=0 $HOME/intarsiaml-hgpk/deploy/tatooine/dev.sh \
  env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=/models/Qwen3.6-35B-A3B-NVFP4 RUST_BACKTRACE=0 \
  cargo test --release -p intarsia-models --test batch_cuda_real -- --ignored real_35b_packed_prefill_equals_the_local_engine --test-threads=1 --nocapture < /dev/null 2>&1 \
  | grep -avE "^plan |MemoryPlan|recurrent bytes" | tail -${TAILN:-25}
