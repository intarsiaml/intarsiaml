#!/usr/bin/env bash
# Sync the working tree to spark1 and build the CLI (+ the CUDA test binaries) there. Runs INSIDE a fleetq slot:
#   fleetq run --host spark1 --mem 0 --label "..." -- scripts/cuda-decode/build.sh <name> [--tests]
# <name> is the tree on spark1: ~/intarsia-decfuse/<name> (base = arch/plan tip, dev = the work). vLLM stays up; the build
# uses few cores (CARGO_BUILD_JOBS, default 6). FLEETQ_JOB (set by `fleetq run`) goes on every ssh.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
NAME=${1:?tree name}; TESTS=${2:-}
HOST=${HOST:-spark1}; JOBS=${CARGO_BUILD_JOBS:-6}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R='~/intarsia-decfuse'
SRC=${SRC_DIR:-$REPO}    # SRC_DIR: a plain export (no .git) of another commit, then INTARSIA_STAMP must be set by the caller
STAMP=${INTARSIA_STAMP:-$("$REPO/scripts/tree-stamp.sh")}
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R/$NAME"
# seed the dependency cache from an earlier tree on the box (registry crates only; workspace crates rebuild)
rsh "test -d $R/$NAME/target || { test -d ~/intarsia-acc35b/src/target && cp -a ~/intarsia-acc35b/src/target $R/$NAME/target; true; }"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude "$SRC/" "$HOST:intarsia-decfuse/$NAME/"
echo "stamp: $STAMP"
rsh "cd $R/$NAME && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=$JOBS CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | tail -5"
if [[ $TESTS == --tests ]]; then
  rsh "cd $R/$NAME && CARGO_BUILD_JOBS=$JOBS CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --no-run 2>&1 | tail -8"
fi
rsh "ls -la $R/$NAME/target/acc/release/intarsia"
