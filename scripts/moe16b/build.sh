#!/usr/bin/env bash
# Lane msd: build the CUDA CLI on spark1 under ~/intarsiaml-moe16b/src, inside a fleetq slot (--load cpu):
#   fleetq run --host spark1 --load cpu --mem 16 --expect 40 --label "msd: build <name>" -- scripts/moe16b/build.sh <name>
# Copies the binary to ~/intarsiaml-moe16b/bin/<name> so base and dev builds coexist (one cargo target, disk is tight).
# REF=<git ref> builds that commit (git archive, e.g. REF=origin/main for the base) instead of the working tree.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
NAME=$1; REPO="$(git rev-parse --show-toplevel)"
HOST=${HOST:-spark1}; SEED=${SEED:-intarsiaml}
R="~/intarsiaml-moe16b/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R ~/intarsiaml-moe16b/bin"  # lane moe16b: a fresh target (spark1 disk is tight: no copy of a seed target)
if [[ -n "${REF:-}" ]]; then
  SHA=$(git -C "$REPO" rev-parse "$REF"); STAMP="ref-${SHA:0:12}"
  rsh "rm -rf ~/intarsiaml-moe16b/stage && mkdir -p ~/intarsiaml-moe16b/stage"
  git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes "$HOST" "tar -x -C ~/intarsiaml-moe16b/stage"
  rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out ~/intarsiaml-moe16b/stage/ $R/ && rm -rf ~/intarsiaml-moe16b/stage"
else
  STAMP="$("$REPO/scripts/tree-stamp.sh")"
  rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees --exclude out "$REPO/" "$HOST:intarsiaml-moe16b/src/"
fi
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
rsh "rm -f $R/target/release/intarsia; cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
rsh "cp $R/target/release/intarsia ~/intarsiaml-moe16b/bin/$NAME; ls -la ~/intarsiaml-moe16b/bin/$NAME; ~/intarsiaml-moe16b/bin/$NAME --version | head -1"
