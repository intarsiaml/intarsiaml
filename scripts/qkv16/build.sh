#!/usr/bin/env bash
# Lane qkv16: sync this worktree to spark1 ~/intarsiaml-qkv16/src and build, inside a fleetq slot (--load cpu):
#   fleetq run --host spark1 --load cpu --mem 16 --expect 30 --label "qkv16: build" -- scripts/qkv16/build.sh <what>...
# what: rows-<name> (the rows_exact test binary -> bin/rows_exact-<name>), cli-<name> (the CUDA CLI -> bin/<name>),
#       breal-<name> (the batch_cuda_real test binary -> bin/batch_cuda_real-<name>),
#       rt-<name> (the row_tables test binary -> bin/row_tables-<name>), qm-<name> (quantmoe -> bin/quantmoe-<name>),
#       fd-<name> (fused_decode -> bin/fused_decode-<name>).
# REF=<git ref> syncs that commit (git archive) instead of the working tree. One cargo target (spark1 disk is tight).
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"; H="$(cd "$(dirname "${BASH_SOURCE[0]}")/../step16" && pwd)"
L='~/intarsiaml-qkv16'; R="$L/src"
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
  rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees --exclude out "$REPO/" "spark1:intarsiaml-qkv16/src/"
fi
# fresh mtimes: rsync -a keeps the Mac's, which can be older than the last build from another tree (cargo would skip it)
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
for what in "$@"; do
  date +%T
  case $what in
    rows-*) rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test rows_exact --no-run 2>&1 | grep -E '^error|Finished' | tail -4; cp \$(ls -t target/release/deps/rows_exact-* | grep -v '\.d$' | head -1) $L/bin/rows_exact-${what#rows-}" ;;
    rt-*) rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test row_tables --no-run 2>&1 | grep -E '^error|Finished' | tail -4; cp \$(ls -t target/release/deps/row_tables-* | grep -v '\.d$' | head -1) $L/bin/row_tables-${what#rt-}" ;;
    qm-*) rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test quantmoe --no-run 2>&1 | grep -E '^error|Finished' | tail -4; cp \$(ls -t target/release/deps/quantmoe-* | grep -v '\.d$' | head -1) $L/bin/quantmoe-${what#qm-}" ;;
    fd-*) rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test fused_decode --no-run 2>&1 | grep -E '^error|Finished' | tail -4; cp \$(ls -t target/release/deps/fused_decode-* | grep -v '\.d$' | head -1) $L/bin/fused_decode-${what#fd-}" ;;
    cli-*) rsh "rm -f $R/target/release/intarsia; cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=16 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -4; cp $R/target/release/intarsia $L/bin/${what#cli-}; $L/bin/${what#cli-} --version | head -1" ;;
    breal-*) rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-models --release --test batch_cuda_real --no-run 2>&1 | grep -E '^error|Finished' | tail -4; cp \$(ls -t target/release/deps/batch_cuda_real-* | grep -v '\.d$' | head -1) $L/bin/batch_cuda_real-${what#breal-}" ;;
  esac
done
rsh "df -h ~ | tail -1; du -sh $L"
date +%T
