#!/usr/bin/env bash
# Lane sp: CUDA no-op (spec off, plan show json/text + 256 ids, 8B and 35B) of the draft-vocab-default branch against origin/main. Copied from scripts/hg/cuda-noop.sh; trees dv-src-base / dv-src-dev (rsynced), built in dv-base / dv-dev.
# branch. Runs ON a Spark inside a fleetq job. Trees: $P/dv-src-base and $P/dv-src-dev (rsynced sources), built in $P/dv-base and
# $P/dv-dev (seeded from an existing built tree, $SEED).
set -uo pipefail
P=$HOME/intarsia-decfuse; O=$P/out/dv; mkdir -p $O
SEED=${SEED:-hg-dev}
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/$(ls /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots | head -1)
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/$(ls /home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots | head -1)
for t in dv-base dv-dev; do
  [ -d $P/$t ] || cp -a $P/$SEED $P/$t
done
echo "== sync"
rsync -rlpgoD --no-times --checksum --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv $P/dv-src-base/ $P/dv-base/
rsync -rlpgoD --no-times --checksum --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv $P/dv-src-dev/ $P/dv-dev/
for t in dv-base dv-dev; do
  echo "== build $t"
  (cd $P/$t && env CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8)
  [ $P/$t/target/acc/release/intarsia -nt $P/$t/crates/intarsia-core/src/plan.rs ] || { echo "BUILD FAILED for $t (binary older than sources)"; exit 1; }
done
for t in base dev; do
  RUN="$P/dv-$t/target/acc/release/intarsia" bash $P/dv-dev/scripts/hg/noop.sh run $O $t cuda $M8 $M35 $P/dv-dev/scripts/hg/prompts --mem-budget 30
done
bash $P/dv-dev/scripts/hg/noop.sh cmp $O base dev
echo "== done"
