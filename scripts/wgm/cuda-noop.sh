#!/usr/bin/env bash
# Lane wg merge gates, ON spark1 inside a fleetq job: CUDA no-op of the merged head (wg-dev) against main (wg-base), knob unset:
# plan show text+json and 256 greedy ids (8B, 35B; short + ~3.5k prompt), `kernels test --quick` on BOTH builds, intarsia-core lib tests on dev.
# Trees $P/wg-src-{base,dev} are rsynced sources (scripts/wgm/sync-cuda.sh on the Mac); builds in $P/wg-{base,dev}, seeded from $SEED.
set -uo pipefail
P=$HOME/intarsia-decfuse; O=$P/out/wg; mkdir -p $O
SEED=${SEED:-hg-dev}
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
H=/home/nvidia/.cache/huggingface/hub
M35=$H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/$(ls $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots | head -1)
M8=$H/models--Qwen--Qwen3-8B/snapshots/$(ls $H/models--Qwen--Qwen3-8B/snapshots | head -1)
for t in wg-base wg-dev; do [ -d $P/$t ] || cp -a $P/$SEED $P/$t; done
echo "== sync $(date +%T)"
for t in base dev; do
  rsync -rlpgoD --no-times --checksum --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv $P/wg-src-$t/ $P/wg-$t/
done
for t in wg-base wg-dev; do
  echo "== build $t $(date +%T)"
  (cd $P/$t && env CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8)
  [ $P/$t/target/acc/release/intarsia -nt $P/$t/crates/intarsia-core/src/plan.rs ] || { echo "BUILD FAILED for $t (binary older than sources)"; exit 1; }
done
for t in base dev; do
  RUN="$P/wg-$t/target/acc/release/intarsia" bash $P/wg-dev/scripts/hg/noop.sh run $O $t cuda $M8 $M35 $P/wg-dev/scripts/hg/prompts --mem-budget 30
done
bash $P/wg-dev/scripts/hg/noop.sh cmp $O base dev
for t in base dev; do
  echo "== kernels test --quick ($t)"
  $P/wg-$t/target/acc/release/intarsia kernels test --quick --backend cuda < /dev/null 2>&1 | tail -n 3
done
echo "== intarsia-core lib tests (dev)"
(cd $P/wg-dev && env CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-core --lib 2>&1 | grep -E "^test result|FAILED|panicked|^error")
echo "== done $(date +%T)"
