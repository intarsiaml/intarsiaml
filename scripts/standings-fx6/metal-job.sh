#!/usr/bin/env zsh
# fx6: Metal, Qwen3-8B BF16, endor. ONE exclusive slot: build the Metal binary from this worktree (separate CARGO_TARGET_DIR),
# then two sameslot.sh runs (scripts/metal-speed-gate) with different visit orders so every engine is visited several times:
#   run 1 (default ABBA-style): intarsia-1 mlx-lm llamacpp intarsia-2
#   run 2 (reversed):           llamacpp mlx-lm intarsia-1 mlx-lm-2 llamacpp-2
# fleetq run --host endor --exclusive --mem 24 --expect 45 --label "fx6 metal standings" -- zsh scripts/standings-fx6/metal-job.sh <out dir>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside fleetq slot" >&2; exit 2; }
OUT=${1:?out dir}
cd ${0:A:h:h:h}
mkdir -p $OUT
export CARGO_TARGET_DIR=$PWD/target-metal
echo "build start $(date)" | tee $OUT/job.log
cargo build --release -p intarsia-cli --features metal 2>&1 | tail -3 | tee -a $OUT/job.log
BIN=$CARGO_TARGET_DIR/release/intarsia
[ -x $BIN ] || { echo "build failed" | tee -a $OUT/job.log; exit 3; }
git rev-parse HEAD > $OUT/commit.txt
echo "run 1 start $(date)" | tee -a $OUT/job.log
ORDER="intarsia-1 mlx-lm llamacpp intarsia-2" zsh scripts/metal-speed-gate/sameslot.sh $BIN $OUT/run1 2>&1 | tee -a $OUT/job.log
echo "run 2 start $(date)" | tee -a $OUT/job.log
ORDER="llamacpp mlx-lm intarsia-1 mlx-lm-2 llamacpp-2" zsh scripts/metal-speed-gate/sameslot.sh $BIN $OUT/run2 2>&1 | tee -a $OUT/job.log
echo "done $(date)" | tee -a $OUT/job.log
