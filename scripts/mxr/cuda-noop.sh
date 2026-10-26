#!/usr/bin/env bash
# Lane mxr (from scripts/mx/cuda-noop.sh): CUDA G0 of the branch against origin/main on spark1, inside a fleetq job: build both
# trees (own target dirs under ~/intarsiaml-mxr-cuda), then plan show + 256 greedy ids (Qwen3-8B, Qwen3.6-35B-A3B NVFP4; short
# and 4k) via scripts/hg/noop.sh, then the comparison and `kernels test --quick` on the branch.
set -uo pipefail
P=$HOME/intarsiaml-mxr-cuda; O=$P/out; mkdir -p $O
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
HUB=/home/nvidia/.cache/huggingface/hub
M35=$HUB/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/$(ls $HUB/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots | head -1)
M8=$HUB/models--Qwen--Qwen3-8B/snapshots/$(ls $HUB/models--Qwen--Qwen3-8B/snapshots | head -1)
for t in base dev; do
  echo "== build $t $(date +%T)"
  (cd $P/src-$t && env CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=$P/target-$t cargo build --release --locked --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8)
  [ $P/target-$t/release/intarsia -nt $P/src-$t/crates/intarsia-core/src/plan.rs ] || { echo "BUILD FAILED for $t (binary older than sources)"; exit 1; }
done
for t in base dev; do
  RUN="$P/target-$t/release/intarsia" bash $P/src-dev/scripts/hg/noop.sh run $O $t cuda $M8 $M35 $P/src-dev/scripts/hg/prompts --mem-budget 30
done
bash $P/src-dev/scripts/hg/noop.sh cmp $O base dev
echo "== kernels test --quick (dev) $(date +%T)"
$P/target-dev/release/intarsia kernels test --quick --backend cuda < /dev/null 2>&1 | tail -3
echo "== done $(date +%T)"
