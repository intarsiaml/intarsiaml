#!/usr/bin/env bash
# Lane hg: HIP gate window, ON tatooine inside a fleetq job that takes radiance down on both cards.
#   A: batching-off no-op, origin/main build (card 0) against the branch build (card 1), in parallel
#   B: card 0: 35B batch_cuda_real (hip) plus the --ignored long-prompt test; card 1: batch_recurrent --test-threads=1, kernels test --quick
# The trap restores radiance whatever happens; the job must only be released after "radiance answers".
set -uo pipefail
O=$HOME/hg-logs/gates; mkdir -p "$O"
D=$HOME/intarsiaml-hg/deploy/tatooine/dev.sh
M8=/models/Qwen3-8B; M35=/models/Qwen3.6-35B-A3B-NVFP4
now() { date '+%F %T'; }
trap 'bash $HOME/hg-win.sh restore' EXIT
echo "[gates] stopping radiance $(now)"
systemctl --user stop radiance
sleep 3
noop() { # noop <tag> <tree> <vol> <gpu>
  RUN="env INTARSIA_REPO=$HOME/intarsiaml-$2 INTARSIA_TARGET_VOL=$3 HIP_VISIBLE_DEVICES=$4 $D /target/release/intarsia" \
    bash $HOME/intarsiaml-hg/scripts/hg/noop.sh run "$O" "$1" hip $M8 $M35 $HOME/intarsiaml-hg/scripts/hg/prompts < /dev/null
}
echo "[gates] A: no-op $(now)"
noop base hg-base intarsia-cargo-target-hgbase 0 > "$O/noop-base.log" 2>&1 &
noop dev hg intarsia-cargo-target-hg 1 > "$O/noop-dev.log" 2>&1 &
wait
cat "$O/noop-base.log" "$O/noop-dev.log"
bash $HOME/intarsiaml-hg/scripts/hg/noop.sh cmp "$O" base dev
echo "[gates] B: tests $(now)"
dev() { # dev <gpu> <cmd...>
  g=$1; shift
  INTARSIA_REPO=$HOME/intarsiaml-hg INTARSIA_TARGET_VOL=intarsia-cargo-target-hg HIP_VISIBLE_DEVICES=$g $D "$@" < /dev/null
}
(
  dev 0 env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 \
    cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture 2>&1 | grep -aE "^test |^test result|FAILED|panicked|^error"
  echo "-- ignored long-prompt"
  dev 0 env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 \
    cargo test --release -p intarsia-models --test batch_cuda_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine --test-threads=1 --nocapture 2>&1 | grep -aE "^test |^test result|FAILED|panicked|^error"
) > "$O/g-c.log" 2>&1 &
(
  dev 1 env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 \
    cargo test --release -p intarsia-models --test batch_recurrent -- --test-threads=1 --nocapture 2>&1 | grep -aE "^test |^test result|FAILED|panicked|^error"
  echo "-- kernels test --quick"
  dev 1 /target/release/intarsia kernels test --backend hip --quick 2>&1 | tail -4
) > "$O/g-r.log" 2>&1 &
wait
echo "== batch_cuda_real (35B, hip)"; cat "$O/g-c.log"
echo "== batch_recurrent (hip) + kernels quick"; cat "$O/g-r.log"
echo "[gates] done $(now)"
