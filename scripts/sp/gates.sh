#!/usr/bin/env bash
# Lane sp, ON tatooine inside a fleetq job that takes radiance down on both cards: the HIP gates of the branch.
#   card 0: 35B batch_cuda_real (hip) + the --ignored long-prompt test
#   card 1: batch_recurrent --test-threads=1, kernels test --quick, rows_exact (incl. the new mxfp4 rows), rows_registry
D=$HOME/intarsiaml-sp/deploy/tatooine/dev.sh
M35=/models/Qwen3.6-35B-A3B-NVFP4
O=$HOME/sp-logs/gates; mkdir -p "$O"
trap 'bash $HOME/sp-win.sh restore' EXIT
systemctl --user stop radiance; sleep 3
dev() { g=$1; shift; INTARSIA_REPO=$HOME/intarsiaml-sp INTARSIA_TARGET_VOL=intarsia-cargo-target-sp HIP_VISIBLE_DEVICES=$g $D "$@" < /dev/null; }
F='^test |^test result|FAILED|panicked|^error|G-B'
(
  dev 0 env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 \
    cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture 2>&1 | grep -aE "$F"
  echo "-- ignored long-prompt"
  dev 0 env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 \
    cargo test --release -p intarsia-models --test batch_cuda_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine --test-threads=1 --nocapture 2>&1 | grep -aE "$F"
) > "$O/g-c.log" 2>&1 &
(
  dev 1 env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 \
    cargo test --release -p intarsia-models --test batch_recurrent -- --test-threads=1 --nocapture 2>&1 | grep -aE "$F"
  echo "-- kernels test --quick"
  dev 1 /target/release/intarsia kernels test --backend hip --quick 2>&1 | tail -4
  echo "-- rows_exact"
  dev 1 env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test rows_exact --test rows_registry -- --test-threads=1 --nocapture 2>&1 | grep -aE "$F"
) > "$O/g-r.log" 2>&1 &
wait
echo "== batch_cuda_real (35B, hip)"; cat "$O/g-c.log"
echo "== batch_recurrent + kernels quick + rows_exact"; cat "$O/g-r.log"
