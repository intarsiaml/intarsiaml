#!/usr/bin/env bash
# Lane tp2: CUDA no-op of the lane head against the lane's base (origin/main 2c2d9941) on spark1, plus the core lib tests, the CUDA
# batching tests and `kernels test --quick` on the head. Runs ON spark1 inside a fleetq job; sources from cuda-sync.sh
# (~/intarsia-decfuse/tp2-src-{base,dev}). Built trees $P/tp2-base, $P/tp2-dev (seeded from an existing built tree, $SEED).
#   scripts/tp2-hip/cuda-sync.sh; fleetq run --host spark1 --mem 40 --expect 60 -- ssh spark1 'bash ~/intarsia-decfuse/tp2-src-dev/scripts/tp2-hip/cuda-noop.sh'
set -uo pipefail
P=$HOME/intarsia-decfuse; O=$P/out/tp2; mkdir -p $O
SEED=${SEED:-hg-base}
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/$(ls /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots | head -1)
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/$(ls /home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots | head -1)
for t in tp2-base tp2-dev; do [ -d $P/$t ] || cp -a $P/$SEED $P/$t; done
echo "== sync"
for t in base dev; do
  rsync -rlpgoD --no-times --checksum --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv $P/tp2-src-$t/ $P/tp2-$t/
done
for t in tp2-base tp2-dev; do
  echo "== build $t"
  (cd $P/$t && env CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8)
  [ $P/$t/target/acc/release/intarsia -nt $P/$t/crates/intarsia-core/src/plan.rs ] || { echo "BUILD FAILED for $t (binary older than sources)"; exit 1; }
done
for t in base dev; do
  RUN="$P/tp2-$t/target/acc/release/intarsia" bash $P/tp2-dev/scripts/tp2-hip/noop/noop.sh run $O $t cuda $M8 $M35 $P/tp2-dev/scripts/tp2-hip/noop/prompts --mem-budget 30
done
bash $P/tp2-dev/scripts/tp2-hip/noop/noop.sh cmp $O base dev
echo "== kernels test --quick (dev)"
$P/tp2-dev/target/acc/release/intarsia kernels test --quick --backend cuda < /dev/null 2>&1 | tail -3
echo "== tests on the head"
cd $P/tp2-dev
env CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-core --lib 2>&1 | grep -E "^test result|FAILED|panicked|^error"
env INTARSIA_BATCH_TEST_BACKEND=cuda INTARSIA_REQUIRE_GPU=1 CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-models --test batch_recurrent -- --test-threads=1 2>&1 | grep -E "^test result|FAILED|panicked|^error"
echo "== done"
