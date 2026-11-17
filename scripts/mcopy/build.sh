#!/usr/bin/env bash
# mcopy: sync this worktree (or REF=<git ref> with OUT=<binary name>) to spark1 ~/intarsiaml-mcopy2/src and build the CUDA CLI. Run under:
#   fleetq run --host spark1 --load cpu --mem 45 --expect 30 --label "mcopy: build" -- bash scripts/mcopy/build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L='~/intarsiaml-mcopy2'; R="$L/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes spark1 "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
# the worktree's tracked files plus its uncommitted changes (the lane edits before it commits)
rsh "mkdir -p $R $L/bin $L/out && rm -rf $L/stage && mkdir -p $L/stage"
# REF=<git ref>: build that commit's tree instead (the baseline binary, OUT=intarsia-main)
if [ -n "${REF:-}" ]; then SRC=(git -C "$REPO" archive --format=tar "$REF"); else SRC=(sh -c 'cd "$1" && COPYFILE_DISABLE=1 git ls-files -z --cached --others --exclude-standard | COPYFILE_DISABLE=1 tar --no-xattrs --null -T - -cf -' _ "$REPO"); fi
OUT=${OUT:-intarsia}
( "${SRC[@]}" ) | ssh -o BatchMode=yes spark1 "tar --warning=no-unknown-keyword -x -C $L/stage"
rsh "find $L/stage -name '._*' -delete; rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
[ -n "${CLEAN_CUDA:-}" ] && rsh "rm -rf $R/target/release/build/intarsia-backend-cuda-* $R/target/release/.fingerprint/intarsia-backend-cuda-*"
STAMP="mcopy2-$(git -C "$REPO" rev-parse --short=12 ${REF:-HEAD})$([ -n "${REF:-}" ] || echo -dirty)"
date +%T
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli > $L/out/build.log 2>&1; grep -E '^(error|warning: unused)|Finished|Compiling intarsia' $L/out/build.log | tail -20; grep -B2 -A25 'failed to run custom build' $L/out/build.log | head -60; cp $R/target/release/intarsia $L/bin/$OUT && $L/bin/$OUT --version | head -1"
rsh "df -h ~ | tail -1; du -sh $L; free -g | sed -n 2p"
date +%T
