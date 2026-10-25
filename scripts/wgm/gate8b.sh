#!/usr/bin/env bash
# Lane wg merge gates, ON tatooine inside wgm-win.sh: the 8B batched==solo test (needs INTARSIA_QWEN3_DIR; the first window did not set it), knob unset and option 1 on.
set -uo pipefail
export HIP_VISIBLE_DEVICES=0
O=$HOME/wgm-logs; mkdir -p "$O"
D=$HOME/intarsiaml-wgm/deploy/tatooine/dev.sh
OPT1="INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all"
for arm in off opt1; do
  e=""; [ $arm = opt1 ] && e="$OPT1"
  echo "-- 8B batched==solo, $arm"
  INTARSIA_REPO=$HOME/intarsiaml-wgm INTARSIA_TARGET_VOL=intarsia-cargo-target-wgm $D env $e INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 \
    INTARSIA_QWEN3_DIR=/models/Qwen3-8B INTARSIA_QWEN36_DIR=/models/Qwen3.6-35B-A3B-NVFP4 \
    cargo test --release --locked -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture real_qwen3_8b 2>&1 < /dev/null | grep -aE "^test |^test result|FAILED|panicked|SKIPPED|^error"
done
echo "[8b] done $(date +%T)"
