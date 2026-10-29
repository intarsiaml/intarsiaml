#!/usr/bin/env bash
# Lane msd: build the CUDA CLI on spark1 under ~/intarsiaml-srv16/src, inside a fleetq slot (--load cpu):
#   fleetq run --host spark1 --load cpu --mem 16 --expect 40 --label "msd: build <name>" -- scripts/srv16/build.sh <name>
# Copies the binary to ~/intarsiaml-srv16/bin/<name> so base and dev builds coexist (one cargo target, disk is tight).
# REF=<git ref> builds that commit (git archive, e.g. REF=origin/main for the base) instead of the working tree.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
NAME=$1; REPO="$(git rev-parse --show-toplevel)"
HOST=${HOST:-spark1}; SEED=${SEED:-intarsiaml}
R="~/intarsiaml-srv16/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R ~/intarsiaml-srv16/bin"  # lane moe16b: a fresh target (spark1 disk is tight: no copy of a seed target)
if [[ -n "${REF:-}" ]]; then
  SHA=$(git -C "$REPO" rev-parse "$REF"); STAMP="ref-${SHA:0:12}"
  rsh "rm -rf ~/intarsiaml-srv16/stage && mkdir -p ~/intarsiaml-srv16/stage"
  git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes "$HOST" "tar -x -C ~/intarsiaml-srv16/stage"
  rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out ~/intarsiaml-srv16/stage/ $R/ && rm -rf ~/intarsiaml-srv16/stage"
else
  STAMP="$("$REPO/scripts/tree-stamp.sh")"
  rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees --exclude out "$REPO/" "$HOST:intarsiaml-srv16/src/"
fi
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
rsh "rm -f $R/target/release/intarsia; cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
rsh "cp $R/target/release/intarsia ~/intarsiaml-srv16/bin/$NAME; ls -la ~/intarsiaml-srv16/bin/$NAME; ~/intarsiaml-srv16/bin/$NAME --version | head -1"
