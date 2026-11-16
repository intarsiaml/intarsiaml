#!/usr/bin/env bash
# itlcreep: sync this worktree to spark1 ~/intarsia-itlcreep/src and build the CUDA CLI. Run under:
#   fleetq run --host spark1 --load cpu --mem 24 --expect 20 --label "itlcreep: build" -- bash scripts/itlcreep/build.sh
# Remove ~/intarsia-itlcreep on spark1 when the lane is done (disk is 92 % full).
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="${REPO:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
L='~/intarsia-itlcreep'; R="$L/src"; BH=spark1
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$BH" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SHA=$(git -C "$REPO" rev-parse "${BUILD_SHA:-ef31cff03}"); STAMP="itlcreep-${SHA:0:12}"  # default SHA: the commit the soak2 binary was built from
rsh "mkdir -p $R $L/bin $L/out"
rsh "rm -rf $L/stage && mkdir -p $L/stage"
git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes "$BH" "tar -x -C $L/stage"
rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
date +%T
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished|Compiling intarsia' | tail -20; cp $R/target/release/intarsia $L/bin/intarsia && $L/bin/intarsia --version | head -1"
rsh "df -h ~ | tail -1; du -sh $L; free -g | sed -n 2p"
date +%T
echo "git sha $SHA"
