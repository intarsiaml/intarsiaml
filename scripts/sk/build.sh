#!/usr/bin/env bash
# Lane sk: build a tree on spark2 under ~/intarsiaml-sk-<name>, inside a fleetq slot (--load cpu):
#   fleetq run --host spark2 --load cpu --mem 16 --expect 25 --label "sk: build <name>" -- scripts/sk/build.sh <name> <src-dir|git-ref>
# <src> is a directory (rsync) or a git ref of this repo (git archive). Cargo target is inside the tree (target/acc), seeded from an
# existing tree's registry cache (SEED) so dependencies are not rebuilt; every synced source is touched afterwards.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
NAME=$1; SRC=$2
HOST=${HOST:-spark2}; SEED=${SEED:-intarsia-decfuse/df-base}
R="~/intarsiaml-sk-$NAME"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
REPO="$(git rev-parse --show-toplevel)"
if [[ -d $SRC ]]; then DIR=$SRC; STAMP="$("$REPO/scripts/tree-stamp.sh")"
else DIR=$(mktemp -d); git -C "$REPO" archive "$SRC" | tar -x -C "$DIR"; STAMP="ref-$(git -C "$REPO" rev-parse --short "$SRC")"; fi
rsh "mkdir -p $R; test -d $R/target || cp -a ~/$SEED/target $R/target"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees "$DIR/" "$HOST:intarsiaml-sk-$NAME/"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
rsh "ls -la $R/target/acc/release/intarsia; $R/target/acc/release/intarsia --version | head -1"
[[ -d $SRC ]] || rm -rf "$DIR"
