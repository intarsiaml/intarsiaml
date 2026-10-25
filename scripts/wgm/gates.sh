#!/usr/bin/env bash
# Lane wg merge gates, ON tatooine inside ONE fleetq job that takes radiance down (both cards), run through wgm-win.sh (restores radiance):
#   A  batching-off / knob-unset no-op: main build (card 0) vs merged-head build (card 1), plan show text+json and 256 greedy ids (8B, 35B, short + 4k)
#   B1 card 1: batch_recurrent --test-threads=1, kernels test --quick (hip), rows_exact + wmma_folds + frag_layout (knob off and opt1 builds)
#   B0 card 0: 35B batch_cuda_real batched==solo ("batched" tests) knob off, then the same with option 1 on, then the --ignored long-prompt test with option 1 on
# usage: gates.sh [A|B|AB]   (default AB). Logs: ~/wgm-logs/.
set -uo pipefail
W=${1:-AB}
O=$HOME/wgm-logs; mkdir -p "$O"
D=$HOME/intarsiaml-wgm/deploy/tatooine/dev.sh
M8=/models/Qwen3-8B; M35=/models/Qwen3.6-35B-A3B-NVFP4
now() { date '+%F %T'; }
OPT1="INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all"
if [[ $W == *A* ]]; then
  noop() { # tag tree vol gpu
    RUN="env INTARSIA_REPO=$HOME/intarsiaml-$2 INTARSIA_TARGET_VOL=$3 HIP_VISIBLE_DEVICES=$4 $D /target/release/intarsia" \
      bash $HOME/intarsiaml-wgm/scripts/hg/noop.sh run "$O/noop" "$1" hip $M8 $M35 $HOME/intarsiaml-wgm/scripts/hg/prompts < /dev/null
  }
  echo "[gates] A no-op $(now)"
  noop base wgm-base intarsia-cargo-target-wgmbase 0 > "$O/noop-base.log" 2>&1 &
  noop dev wgm intarsia-cargo-target-wgm 1 > "$O/noop-dev.log" 2>&1 &
  wait
  cat "$O/noop-base.log" "$O/noop-dev.log"
  bash $HOME/intarsiaml-wgm/scripts/hg/noop.sh cmp "$O/noop" base dev | tee "$O/noop-cmp.txt"
fi
if [[ $W == *B* ]]; then
  dev() { g=$1; shift; INTARSIA_REPO=$HOME/intarsiaml-wgm INTARSIA_TARGET_VOL=intarsia-cargo-target-wgm HIP_VISIBLE_DEVICES=$g $D "$@" < /dev/null; }
  T="env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35"
  echo "[gates] B $(now)"
  (
    echo "-- batch_recurrent"
    dev 1 $T cargo test --release --locked -p intarsia-models --test batch_recurrent -- --test-threads=1 --nocapture 2>&1 | grep -aE "^test |^test result|FAILED|panicked|^error"
    echo "-- kernels test --quick (hip)"
    dev 1 /target/release/intarsia kernels test --backend hip --quick 2>&1 | tail -n 4
    echo "-- rows_exact + wmma_folds + frag_layout, knob unset"
    dev 1 env INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_DEVICE=0 cargo test --release --locked -p intarsia-backend-hip --test rows_exact --test wmma_folds --test frag_layout --no-fail-fast -- --test-threads=1 2>&1 | grep -aE "^test result|FAILED|panicked"
    echo "-- same, option 1 on"
    dev 1 env $OPT1 INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_DEVICE=0 cargo test --release --locked -p intarsia-backend-hip --test rows_exact --test wmma_folds --test frag_layout --no-fail-fast -- --test-threads=1 2>&1 | grep -aE "^test result|FAILED|panicked"
  ) > "$O/g-r.log" 2>&1 &
  (
    echo "-- 35B/8B batched==solo, knob unset"
    dev 0 $T cargo test --release --locked -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture batched 2>&1 | grep -aE "^test |^test result|FAILED|panicked|^error"
    echo "-- 35B/8B batched==solo, option 1 ON ($OPT1)"
    dev 0 env $OPT1 $T cargo test --release --locked -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture batched 2>&1 | grep -aE "^test |^test result|FAILED|panicked|^error"
    echo "-- ignored long-prompt, option 1 ON"
    dev 0 env $OPT1 $T cargo test --release --locked -p intarsia-models --test batch_cuda_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine --test-threads=1 --nocapture 2>&1 | grep -aE "^test |^test result|FAILED|panicked|^error"
  ) > "$O/g-c.log" 2>&1
  wait
  echo "== batch_cuda_real"; cat "$O/g-c.log"
  echo "== batch_recurrent + kernels quick + rows_exact"; cat "$O/g-r.log"
fi
echo "[gates] done $(now)"
