#!/usr/bin/env bash
# Lane mx (copy of scripts/hg/cuda-noop.sh with mx trees): CUDA no-op of the branch against origin/main, plus the core lib tests and CUDA batching tests on the
# branch. Runs ON spark1 inside a fleetq job (sources from scripts/mx/cuda-sync.sh). Trees: $P/mx-src-base and $P/mx-src-dev, built
# $P/mx-dev (seeded from an existing built tree, $SEED).
set -uo pipefail
P=$HOME/intarsia-decfuse; O=$P/out/mx; mkdir -p $O
SEED=${SEED:-hg-base}
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/$(ls /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots | head -1)
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/$(ls /home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots | head -1)
for t in mx-base mx-dev; do
  [ -d $P/$t ] || cp -a $P/$SEED $P/$t
done
echo "== sync"
rsync -rlpgoD --no-times --checksum --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv $P/mx-src-base/ $P/mx-base/
rsync -rlpgoD --no-times --checksum --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv $P/mx-src-dev/ $P/mx-dev/
for t in mx-base mx-dev; do
  echo "== build $t"
  (cd $P/$t && env CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8)
  [ $P/$t/target/acc/release/intarsia -nt $P/$t/crates/intarsia-core/src/plan.rs ] || { echo "BUILD FAILED for $t (binary older than sources)"; exit 1; }
done
for t in base dev; do
  RUN="$P/mx-$t/target/acc/release/intarsia" bash $P/mx-dev/scripts/hg/noop.sh run $O $t cuda $M8 $M35 $P/mx-dev/scripts/hg/prompts --mem-budget 30
done
bash $P/mx-dev/scripts/hg/noop.sh cmp $O base dev
echo "== kernels test --quick (dev)"
$P/mx-dev/target/acc/release/intarsia kernels test --quick --backend cuda < /dev/null 2>&1 | tail -3
echo "== tests on the branch"
cd $P/mx-dev
env CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-core --lib 2>&1 | grep -E "^test result|FAILED|panicked|^error"
env INTARSIA_BATCH_TEST_BACKEND=cuda INTARSIA_REQUIRE_GPU=1 CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-models --test batch_recurrent -- --test-threads=1 2>&1 | grep -E "^test result|FAILED|panicked|^error"
echo "== done"
