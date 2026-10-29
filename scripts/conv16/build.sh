#!/usr/bin/env bash
# Lane conv16: sync this worktree to spark1 ~/intarsiaml-conv16/src and build, inside a fleetq slot (--load cpu):
#   fleetq run --host spark1 --load cpu --mem 16 --expect 30 --label "conv16: build" -- scripts/conv16/build.sh <what>...
# what: rows-<name> (the rows_exact test binary -> bin/rows_exact-<name>), cli-<name> (the CUDA CLI -> bin/<name>),
#       breal-<name> (the batch_cuda_real test binary -> bin/batch_cuda_real-<name>),
#       rt-<name> (the row_tables test binary -> bin/row_tables-<name>), gs-<name> (gdn_split -> bin/gdn_split-<name>), c16-<name> (gdn_conv16 -> bin/gdn_conv16-<name>), sass (instruction counts of bin/rows_exact-base and -dev).
# REF=<git ref> syncs that commit (git archive) instead of the working tree. One cargo target (spark1 disk is tight).
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"; H="$(cd "$(dirname "${BASH_SOURCE[0]}")/../step16" && pwd)"
L='~/intarsiaml-conv16'; R="$L/src"
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
  rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees --exclude out "$REPO/" "spark1:intarsiaml-conv16/src/"
fi
# fresh mtimes: rsync -a keeps the Mac's, which can be older than the last build from another tree (cargo would skip it)
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
for what in "$@"; do
  date +%T
  case $what in
    rows-*) rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test rows_exact --no-run 2>&1 | grep -E '^error|Finished' | tail -4; cp \$(ls -t target/release/deps/rows_exact-* | grep -v '\.d$' | head -1) $L/bin/rows_exact-${what#rows-}" ;;
    rt-*) rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test row_tables --no-run 2>&1 | grep -E '^error|Finished' | tail -4; cp \$(ls -t target/release/deps/row_tables-* | grep -v '\.d$' | head -1) $L/bin/row_tables-${what#rt-}" ;;
    c16-*) rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test gdn_conv16 --no-run 2>&1 | grep -E '^error|Finished' | tail -4; cp \$(ls -t target/release/deps/gdn_conv16-* | grep -v '\.d$' | head -1) $L/bin/gdn_conv16-${what#c16-}" ;;
    ps-*) rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-models --release --test prefix_store_cuda --no-run 2>&1 | grep -E '^error|Finished' | tail -4; cp \$(ls -t target/release/deps/prefix_store_cuda-* | grep -v '\.d$' | head -1) $L/bin/prefix_store_cuda-${what#ps-}" ;;
    gs-*) rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test gdn_split --no-run 2>&1 | grep -E '^error|Finished' | tail -4; cp \$(ls -t target/release/deps/gdn_split-* | grep -v '\.d$' | head -1) $L/bin/gdn_split-${what#gs-}" ;;
    cli-*) rsh "rm -f $R/target/release/intarsia; cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=16 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -4; cp $R/target/release/intarsia $L/bin/${what#cli-}; $L/bin/${what#cli-} --version | head -1" ;;
    breal-*) rsh "cd $R && CARGO_BUILD_JOBS=16 cargo test -p intarsia-models --release --test batch_cuda_real --no-run 2>&1 | grep -E '^error|Finished' | tail -4; cp \$(ls -t target/release/deps/batch_cuda_real-* | grep -v '\.d$' | head -1) $L/bin/batch_cuda_real-${what#breal-}" ;;
    chk) rsh "cd $L/bin && for f in *; do printf '%s INTARSIA_CUDA_CONV_BF16 strings: ' \$f; grep -ac INTARSIA_CUDA_CONV_BF16 \$f; done" ;;
    sass) scp -q "$H/sass.sh" spark1:intarsiaml-conv16/sass.sh; rsh "bash $L/sass.sh $L/bin/rows_exact-base $L/bin/rows_exact-dev" ;;
  esac
done
rsh "df -h ~ | tail -1; du -sh $L"
date +%T
