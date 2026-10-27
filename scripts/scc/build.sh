#!/usr/bin/env bash
# Lane scc: build the CUDA CLI on spark2 under ~/intarsiaml-scc/src, inside a fleetq slot (--load cpu):
#   fleetq run --host spark2 --load cpu --mem 16 --expect 40 --label "scc: build <name>" -- scripts/scc/build.sh <name>
# Copies the binary to ~/intarsiaml-scc/bin/<name> so base and dev builds coexist (one cargo target, disk is tight).
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
NAME=$1; REPO="$(git rev-parse --show-toplevel)"
HOST=${HOST:-spark2}; SEED=${SEED:-intarsiaml-pc}
R="~/intarsiaml-scc/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
STAMP="$("$REPO/scripts/tree-stamp.sh")"
rsh "mkdir -p $R ~/intarsiaml-scc/bin; test -d $R/target || cp -a ~/$SEED/target $R/target"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees --exclude out "$REPO/" "$HOST:intarsiaml-scc/src/"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
rsh "cp $R/target/release/intarsia ~/intarsiaml-scc/bin/$NAME; ls -la ~/intarsiaml-scc/bin/$NAME; ~/intarsiaml-scc/bin/$NAME --version | head -1"
