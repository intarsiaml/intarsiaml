#!/usr/bin/env bash
# Lane rip2 (copied from scripts/rip): build the CUDA CLI and the GPU test binaries (batch_cuda_real, row_invariance, rows_exact) on spark2 under
# ~/intarsiaml-rip2/src (inside a fleetq slot, --load cpu):
#   fleetq run --host spark2 --load cpu --mem 8 --expect 40 --label "rip2: build" -- scripts/rip2/build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
H=${HOST:-spark2}; R="~/intarsiaml-rip2/src"
RPATH='set -o pipefail; export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$H" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
STAMP="$("$REPO/scripts/tree-stamp.sh")"
rsh "mkdir -p $R ~/intarsiaml-rip2/bin ~/intarsiaml-rip2/out; test -d $R/target || cp -a ~/intarsiaml-pc/target $R/target"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees --exclude out "$REPO/" "$H:intarsiaml-rip2/src/"
# rsync -a keeps the Mac mtimes: a file edited while an earlier build compiled can look older than its fingerprint (lane rip2's
# stale binary), so touch the sources as build-base.sh does
rsh "cd $R && find . -path ./target -prune -o -type f \\( -name '*.rs' -o -name '*.cu' -o -name '*.h' -o -name '*.cuh' -o -name '*.toml' \\) -print0 | xargs -0 touch"
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=16 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
rsh "cp $R/target/release/intarsia ~/intarsiaml-rip2/bin/intarsia; ~/intarsiaml-rip2/bin/intarsia --version | head -1"
rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-models --release --test batch_cuda_real --no-run 2>&1 | grep -E '^error|Finished|Executable' | tail -4"
rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test row_invariance --test rows_exact --no-run 2>&1 | grep -E '^error|Finished|Executable' | tail -4"
echo "stamp $STAMP"
