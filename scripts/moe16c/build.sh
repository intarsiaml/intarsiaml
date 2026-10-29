#!/usr/bin/env bash
# Lane moe16c: sync this worktree to spark1 ~/intarsiaml-moe16c/src and build, inside a fleetq slot (--load cpu):
#   fleetq run --host spark1 --load cpu --mem 16 --expect 30 --label "moe16c: build" -- scripts/moe16c/build.sh <what>...
# what: rows-<name> (the rows_exact test binary -> bin/rows_exact-<name>), cli-<name> (the CUDA CLI -> bin/<name>),
#       sass (instruction counts of bin/rows_exact-base and -dev).
# REF=<git ref> syncs that commit (git archive) instead of the working tree. One cargo target (spark1 disk is tight).
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"; H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L='~/intarsiaml-moe16c'; R="$L/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes spark1 "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R $L/bin"
if [[ -n "${REF:-}" ]]; then
  SHA=$(git -C "$REPO" rev-parse "$REF"); STAMP="ref-${SHA:0:12}"
  rsh "rm -rf $L/stage && mkdir -p $L/stage"
  git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes spark1 "tar -x -C $L/stage"
  rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
else
  STAMP="$("$REPO/scripts/tree-stamp.sh")"
  rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees --exclude out "$REPO/" "spark1:intarsiaml-moe16c/src/"
fi
# fresh mtimes: rsync -a keeps the Mac's, which can be older than the last build from another tree (cargo would skip it)
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
for what in "$@"; do
  date +%T
  case $what in
    rows-*) rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test rows_exact --no-run 2>&1 | grep -E '^error|Finished' | tail -4; cp \$(ls -t target/release/deps/rows_exact-* | grep -v '\.d$' | head -1) $L/bin/rows_exact-${what#rows-}" ;;
    cli-*) rsh "rm -f $R/target/release/intarsia; cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=16 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -4; cp $R/target/release/intarsia $L/bin/${what#cli-}; $L/bin/${what#cli-} --version | head -1" ;;
    sass) scp -q "$H/sass.sh" spark1:intarsiaml-moe16c/sass.sh; rsh "bash $L/sass.sh $L/bin/rows_exact-base $L/bin/rows_exact-dev" ;;
  esac
done
rsh "df -h ~ | tail -1; du -sh $L"
date +%T
