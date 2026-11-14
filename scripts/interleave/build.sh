#!/usr/bin/env bash
# lane interleave: build the CUDA CLI of commit $SHA (default: this worktree's HEAD) on $BHOST into $ABL (scripts/ab35c build.sh with a
# commit argument, so the main arm builds without a second worktree). Run under: fleetq run --host "$BHOST" --load cpu ...
#   SHA=10b235785 BHOST=spark2 ABL='~/intarsia-il-main' fleetq run --host spark2 --load cpu --mem 24 --expect 25 -- bash scripts/interleave/build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L="${ABL:?ABL: build dir on the host, e.g. ~/intarsia-interleave}"; R="$L/src"; H="${BHOST:?BHOST}"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$H" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SHA=$(git -C "$REPO" rev-parse "${SHA:-HEAD}^{commit}"); STAMP="interleave-${SHA:0:12}"
rsh "mkdir -p $R $L/bin $L/out"
# a target dir of a recent build on the same host saves the dependencies (SEED, optional)
[[ -n "${SEED:-}" ]] && rsh "test -d $R/target || cp -a $SEED $R/target"
rsh "rm -rf $L/stage && mkdir -p $L/stage"
git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes "$H" "tar -x -C $L/stage"
rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
date +%T
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished|Compiling intarsia' | tail -20; cp $R/target/release/intarsia $L/bin/intarsia && $L/bin/intarsia --version | head -1"
rsh "df -h ~ | tail -1; du -sh $L; free -g | sed -n 2p"
date +%T
echo "git sha $SHA"
