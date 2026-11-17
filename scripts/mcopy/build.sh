#!/usr/bin/env bash
# mcopy: sync this worktree to spark1 ~/intarsiaml-mcopy/src and build the CUDA CLI. Run under:
#   fleetq run --host spark1 --load cpu --mem 45 --expect 30 --label "mcopy: build" -- bash scripts/mcopy/build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L='~/intarsiaml-mcopy'; R="$L/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes spark1 "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
# the worktree's tracked files plus its uncommitted changes (the lane edits before it commits)
rsh "mkdir -p $R $L/bin $L/out && rm -rf $L/stage && mkdir -p $L/stage"
( cd "$REPO" && COPYFILE_DISABLE=1 git ls-files -z --cached --others --exclude-standard | COPYFILE_DISABLE=1 tar --no-xattrs --null -T - -cf - ) | ssh -o BatchMode=yes spark1 "tar --warning=no-unknown-keyword -x -C $L/stage"
rsh "find $L/stage -name '._*' -delete; rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
[ -n "${CLEAN_CUDA:-}" ] && rsh "rm -rf $R/target/release/build/intarsia-backend-cuda-* $R/target/release/.fingerprint/intarsia-backend-cuda-*"
STAMP="mcopy-$(git -C "$REPO" rev-parse --short=12 HEAD)-dirty"
rsh "[ -x $L/bin/intarsia ] && cp -f $L/bin/intarsia $L/bin/intarsia-prev || true"   # the binary of the previous build, for A/B against it
date +%T
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli > $L/out/build.log 2>&1; grep -E '^(error|warning: unused)|Finished|Compiling intarsia' $L/out/build.log | tail -20; grep -B2 -A25 'failed to run custom build' $L/out/build.log | head -60; cp $R/target/release/intarsia $L/bin/intarsia && $L/bin/intarsia --version | head -1"
rsh "df -h ~ | tail -1; du -sh $L; free -g | sed -n 2p"
date +%T
