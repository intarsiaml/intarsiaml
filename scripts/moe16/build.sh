#!/usr/bin/env bash
# Lane moe16: build the CUDA CLI on spark1 under ~/intarsiaml-moe16/src from this worktree, inside a fleetq slot (--load cpu):
#   fleetq run --host spark1 --load cpu --mem 16 --expect 45 --label "moe16: build" -- scripts/moe16/build.sh <name>
# Fresh cargo target (no copy of another checkout's target: spark1's disk is tight); the binary goes to ~/intarsiaml-moe16/bin/<name>.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
NAME=$1; REPO="$(git rev-parse --show-toplevel)"; HOST=${HOST:-spark1}
R="~/intarsiaml-moe16/src"
rsh() { ssh -n -o BatchMode=yes "$HOST" "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB; $*"; }
STAMP="$("$REPO/scripts/tree-stamp.sh")"
rsh "mkdir -p $R ~/intarsiaml-moe16/bin ~/intarsiaml-moe16/out"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees \
  --exclude .scratch-vllm --exclude out "$REPO/" "$HOST:intarsiaml-moe16/src/"
rsh "rm -f $R/target/release/intarsia; cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
rsh "cp $R/target/release/intarsia ~/intarsiaml-moe16/bin/$NAME; ls -la ~/intarsiaml-moe16/bin/$NAME; ~/intarsiaml-moe16/bin/$NAME --version | head -1; df -h ~ | tail -1; du -sh $R/target"
