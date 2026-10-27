#!/usr/bin/env bash
# Lane cgv: build a git rev (or the working tree) on spark1 under ~/intarsiaml-cgv/trees/<name>, inside a fleetq slot (--load cpu):
#   fleetq run --host spark1 --load cpu --mem 16 --expect 25 --label "cgv: build <name>" -- scripts/cgv/build.sh <name> [rev|WT]
# rev = a git rev exported with git archive (default WT = this worktree, rsync). The cargo target is seeded from $SEED
# (default ~/intarsiaml-cgv/trees/main/target when it exists, else ~/intarsiaml-m1/target).
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
NAME=$1; REV=${2:-WT}; REPO="$(git rev-parse --show-toplevel)"
HOST=${HOST:-spark1}
R="intarsiaml-cgv/trees/$NAME"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p ~/$R; test -d ~/$R/target || { if test -d ~/intarsiaml-cgv/trees/main/target; then cp -a ~/intarsiaml-cgv/trees/main/target ~/$R/target; else cp -a ~/intarsiaml-m1/target ~/$R/target; fi; }"
if [[ $REV == WT ]]; then
  STAMP="$("$REPO/scripts/tree-stamp.sh")"
  rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees --exclude out "$REPO/" "$HOST:$R/"
else
  STAMP="rev=$(git rev-parse --short "$REV")"
  git -C "$REPO" archive "$REV" | ssh -o BatchMode=yes "$HOST" "cd ~/$R && find . -mindepth 1 -maxdepth 1 ! -name target ! -name out -exec rm -rf {} + && tar -x"
fi
rsh "cd ~/$R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
rsh "cd ~/$R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-12} cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
rsh "ls -la ~/$R/target/release/intarsia; ~/$R/target/release/intarsia --version | head -1"
