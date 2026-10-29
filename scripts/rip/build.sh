#!/usr/bin/env bash
# Lane rip: build the CUDA CLI and the GPU test binaries (batch_cuda_real, row_invariance, rows_exact) on spark2 under
# ~/intarsiaml-rip/src (inside a fleetq slot, --load cpu):
#   fleetq run --host spark2 --load cpu --mem 8 --expect 40 --label "rip: build" -- scripts/rip/build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
H=${HOST:-spark2}; R="~/intarsiaml-rip/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$H" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
STAMP="$("$REPO/scripts/tree-stamp.sh")"
rsh "mkdir -p $R ~/intarsiaml-rip/bin ~/intarsiaml-rip/out; test -d $R/target || cp -a ~/intarsiaml-pc/target $R/target"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees --exclude out "$REPO/" "$H:intarsiaml-rip/src/"
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=16 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
rsh "cp $R/target/release/intarsia ~/intarsiaml-rip/bin/intarsia; ~/intarsiaml-rip/bin/intarsia --version | head -1"
rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-models --release --test batch_cuda_real --no-run 2>&1 | grep -E '^error|Finished|Executable' | tail -4"
rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test row_invariance --test rows_exact --no-run 2>&1 | grep -E '^error|Finished|Executable' | tail -4"
echo "stamp $STAMP"
