#!/usr/bin/env bash
# Lane cmx: build a tree on spark1 under ~/intarsiaml-cmx/trees/<name>, inside a fleetq slot (--load cpu):
#   fleetq run --host spark1 --load cpu --mem 16 --expect 25 --label "cmx: build <name>" -- scripts/cmx/build.sh <name> [src-dir]
# Same as scripts/s27d/build.sh with other paths; the cargo target is seeded from the s27d dev tree.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
NAME=$1; REPO="$(git rev-parse --show-toplevel)"; SRC=${2:-$REPO}
HOST=${HOST:-spark1}; SEED=${SEED:-intarsiaml-s27d/trees/dev}
R="~/intarsiaml-cmx/trees/$NAME"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
STAMP="$("$REPO/scripts/tree-stamp.sh")"
rsh "mkdir -p $R; test -d $R/target || cp -a ~/$SEED/target $R/target"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees --exclude out "$SRC/" "$HOST:intarsiaml-cmx/trees/$NAME/"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-12} cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
rsh "ls -la $R/target/release/intarsia; $R/target/release/intarsia --version | head -1"
