#!/usr/bin/env bash
# dsubset: sync this worktree's HEAD to spark1 ~/intarsia-dsubset/src and build the CUDA CLI (seeded from another lane's target dir).
# REV: the git revision to build (default HEAD). OUTBIN: the binary's name under ~/intarsia-dsubset/bin (default intarsia).
# Run under: fleetq run --host spark1 --load cpu --mem 34 --expect 30 --label "dsubset: build" -- scripts/dsubset/build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
H=spark1; L='~/intarsia-dsubset'; R="$L/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes $H "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SHA=$(git -C "$REPO" rev-parse "${REV:-HEAD}"); STAMP="dsubset-${SHA:0:12}"
rsh "mkdir -p $R $L/bin $L/out"
rsh "[ -d $R/target ] || { [ -d ~/intarsia-interleave/src/target ] && cp -a ~/intarsia-interleave/src/target $R/target; true; }"
rsh "rm -rf $L/stage && mkdir -p $L/stage"
git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes $H "tar -x -C $L/stage"
rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
scp -q -o BatchMode=yes "$REPO/scripts/dsubset/memwatch.sh" $H:intarsia-dsubset/memwatch.sh  # from this worktree (REV may predate it)
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
date +%T
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished|Compiling intarsia' | tail -20; cp $R/target/release/intarsia $L/bin/${OUTBIN:-intarsia} && $L/bin/${OUTBIN:-intarsia} --version | head -1"
rsh "df -h ~ | tail -1; du -sh $L; free -g | sed -n 2p; systemctl is-active earlyoom"
date +%T
echo "git sha $SHA"
