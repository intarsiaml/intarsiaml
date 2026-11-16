#!/bin/sh
# lazyc: sync this worktree to spark1 ~/intarsia-lazyc/src and build the CUDA CLI. Run under: fleetq run --host spark1 --load cpu ...
set -eu
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
L='~/intarsia-lazyc'; R="$L/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes spark1 "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
STAMP="lazyc-$(git -C "$REPO" rev-parse --short=12 HEAD)"
rsh "mkdir -p $R $L/bin $L/out && rm -rf $L/stage && mkdir -p $L/stage"
# the working tree (it carries the uncommitted flip), not a commit
( cd "$REPO" && COPYFILE_DISABLE=1 tar --exclude=./target --exclude=./.worktrees --exclude=./.git --exclude=./golden --exclude=./tools/.venv -cf - . ) | ssh -o BatchMode=yes spark1 "tar -x -C $L/stage"
rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage && find $R -name '._*' -type f -not -path '*/target/*' -delete"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
date +%T
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli > $L/out/build.log 2>&1; tail -30 $L/out/build.log; cp $R/target/release/intarsia $L/bin/intarsia && $L/bin/intarsia --version | head -1"
date +%T
