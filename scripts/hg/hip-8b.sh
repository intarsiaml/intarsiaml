#!/usr/bin/env bash
# Lane hg: the 8B batched-decode HIP test (needs INTARSIA_QWEN3_DIR, which the first gate run left unset), radiance down for its duration.
set -uo pipefail
trap 'bash $HOME/hg-win.sh restore' EXIT
systemctl --user stop radiance
sleep 3
INTARSIA_REPO=$HOME/intarsiaml-hg INTARSIA_TARGET_VOL=intarsia-cargo-target-hg HIP_VISIBLE_DEVICES=0 $HOME/intarsiaml-hg/deploy/tatooine/dev.sh \
  env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN3_DIR=/models/Qwen3-8B \
  cargo test --release -p intarsia-models --test batch_cuda_real real_qwen3_8b -- --test-threads=1 --nocapture < /dev/null 2>&1 \
  | grep -aE "^test |^test result|FAILED|panicked|^error"
