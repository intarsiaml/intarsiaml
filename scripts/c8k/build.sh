#!/usr/bin/env bash
# Lane c8k (from scripts/lt/build.sh): sync a source dir to a Spark, build the CUDA CLI, optionally set the CUDA prefill chunk of
# the 35B recipe in the Spark copy only (RECIPE_CHUNK=8192: the "recipe override" for the cargo tests; the repo's recipe is untouched)
# and build the intarsia-models test binaries (TESTS=1). Run inside a fleetq slot (--load cpu):
#   FLEETQ_OWNER=claude-f31ccf91-c8k HOST=spark1 fleetq run --host spark1 --load cpu --expect 30 --label "c8k build" -- scripts/c8k/build.sh <tree> /opt/code/wt-c8k [stamp]
# Trees live in ~/intarsia-decfuse/<tree> on HOST; the target dir is seeded from SEED_TREE (default lt-main).
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=${HOST:-spark1}; T=$1; SRC=$2; STAMP=${3:-auto}
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The build stamp must be `scripts/tree-stamp.sh` output (the baseline gate's provenance check parses it). Anything else: engine
# source from the clean main worktree CLEAN_SRC (no scripts/c8k in the code tree), stamped from there.
if [[ $STAMP != head=* ]]; then
  SRC=${CLEAN_SRC:-/opt/code/wt-c8k-src}; STAMP=$("$SRC/scripts/tree-stamp.sh")
fi
[[ -n ${RECIPE_CHUNK:-} ]] && STAMP=${STAMP/dirty=no/dirty=yes}
SEED=${SEED_TREE:-lt-main}; R='~/intarsia-decfuse'
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RP FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R/$T; if [ ! -d $R/$T/target ] && [ -d $R/$SEED/target ]; then cp -a $R/$SEED/target $R/$T/target; fi"
rsync -a --delete --exclude target --exclude 'target-*' --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude docs/results "$SRC/" "$HOST:intarsia-decfuse/$T/"
if [[ -n ${RECIPE_CHUNK:-} ]]; then
  rsh "cd $R/$T && sed -i 's/^prefill_chunk = 2048\$/prefill_chunk = $RECIPE_CHUNK/' recipes/qwen3_5_moe.toml && grep -n '^prefill_chunk' recipes/qwen3_5_moe.toml"
fi
rsh "cd $R/$T && find . -path ./target -prune -o -type f \\( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \\) -print0 | xargs -0 touch"
rsh "cd $R/$T && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
rsh "ls -la $R/$T/target/acc/release/intarsia"
if [[ -n ${TESTS:-} ]]; then
  rsh "cd $R/$T && [ -d target/test ] || cp -a target/acc target/test; env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} CARGO_TARGET_DIR=target/test cargo test --release -p intarsia-models --test batch_cuda_real --no-run 2>&1 | grep -E '^(error)|Finished|Executable' | tail -8"
fi
# helper scripts and the 35B goldens / oracles for the gate runs
ssh -n -o BatchMode=yes "$HOST" "mkdir -p ~/intarsia-decfuse/pf-scripts ~/intarsia-decfuse/pf-golden"
rsync -a "$D/" "$HOST:intarsia-decfuse/pf-scripts/c8k/"
rsync -a "$D/../lt/" "$HOST:intarsia-decfuse/pf-scripts/lt/"
rsync -a --include='qwen36*' --include='oracle/***' --include='ref_noise/***' --include='baselines/***' --exclude='*' "$D/../../golden/" "$HOST:intarsia-decfuse/pf-golden/"
