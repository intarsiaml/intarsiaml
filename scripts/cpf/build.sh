#!/usr/bin/env bash
# Lane pf (cuda-pack-flip; scripts/cpf, scripts/pf is an older lane): sync this worktree to spark1 ~/intarsiaml-pf/src and build, inside a fleetq slot (--load cpu):
#   fleetq run --host spark1 --load cpu --mem 16 --expect 60 --label "pf: build" -- scripts/cpf/build.sh [cli|test|all] [bin-name]
# cli: release CUDA CLI, copied to ~/intarsiaml-pf/bin/<bin-name>; test: the batch_cuda_real and CUDA packed_rows test binaries (no run).
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
WHAT=${1:-all}; NAME=${2:-dev}; REPO="$(git rev-parse --show-toplevel)"
HOST=${HOST:-spark1}; R="~/intarsiaml-pf/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
STAMP="$("$REPO/scripts/tree-stamp.sh")"
rsh "mkdir -p $R ~/intarsiaml-pf/bin ~/intarsiaml-pf/out; test -d $R/target || cp -a ~/${SEED:-intarsiaml-bm/src}/target $R/target"
# rsync keeps the Mac's mtimes, which can be older than the (seeded) target's artifacts: touch every file it transferred
FIRST=$(ssh -n -o BatchMode=yes "$HOST" "test -f intarsiaml-pf/src/.synced || echo 1")
rsync -a --delete --out-format='%n' --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees --exclude .scratch --exclude out --exclude .synced "$REPO/" "$HOST:intarsiaml-pf/src/" | grep -v '/$' > "$REPO/.scratch/synced.txt" || true
if [[ -n $FIRST ]]; then
  rsh "cd $R && find . -path ./target -prune -o -type f -print0 | xargs -0 touch && touch .synced"
elif [[ -s $REPO/.scratch/synced.txt ]]; then
  ssh -o BatchMode=yes "$HOST" "cd intarsiaml-pf/src && xargs -d '\n' touch --" < "$REPO/.scratch/synced.txt"
fi
echo "synced $(wc -l < "$REPO/.scratch/synced.txt") files"
if [[ $WHAT == cli || $WHAT == all ]]; then
  rsh "rm -f $R/target/release/intarsia; cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
  rsh "cp $R/target/release/intarsia ~/intarsiaml-pf/bin/$NAME; ~/intarsiaml-pf/bin/$NAME --version | head -1"
fi
if [[ $WHAT == test || $WHAT == all ]]; then
  rsh "cd $R && CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} cargo test -p intarsia-models --release --test batch_cuda_real --no-run 2>&1 | grep -E '^error|Finished|Executable' -A3 | tail -8"
  rsh "cd $R && CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} cargo test -p intarsia-backend-cuda --release --test packed_rows --no-run 2>&1 | grep -E '^error|Finished|Executable' -A3 | tail -8"
fi
echo "BUILD DONE $WHAT $NAME"
