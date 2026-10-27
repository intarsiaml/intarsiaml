#!/usr/bin/env bash
# Lane vr16: build the CUDA CLI on spark2 under ~/intarsiaml-vr16/src, inside a fleetq slot (--load cpu):
#   fleetq run --host spark2 --load cpu --mem 16 --expect 40 --label "scc: build <name>" -- scripts/vr16/build.sh <name>
# Copies the binary to ~/intarsiaml-vr16/bin/<name> so base and dev builds coexist (one cargo target, disk is tight).
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
NAME=$1; REPO="$(git rev-parse --show-toplevel)"
HOST=${HOST:-spark2}; SEED=${SEED:-intarsiaml-pc}
R="~/intarsiaml-vr16/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
STAMP="$("$REPO/scripts/tree-stamp.sh")"
rsh "mkdir -p $R ~/intarsiaml-vr16/bin; test -d $R/target || cp -a ~/$SEED/target $R/target"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees --exclude out "$REPO/" "$HOST:intarsiaml-vr16/src/"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
rsh "cp $R/target/release/intarsia ~/intarsiaml-vr16/bin/$NAME; ls -la ~/intarsiaml-vr16/bin/$NAME; ~/intarsiaml-vr16/bin/$NAME --version | head -1"
