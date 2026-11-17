#!/usr/bin/env bash
# herd: sync this worktree to spark2 ~/intarsiaml-herd/src and build the CUDA CLI. Run under: fleetq run --host spark2 --load cpu ...
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOSTN=${HOSTN:-spark2}
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L='~/intarsiaml-herd'; R="$L/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOSTN" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SHA=$(git -C "$REPO" rev-parse HEAD); STAMP="herd-${SHA:0:12}"
rsh "mkdir -p $R $L/bin $L/out"
# warm start: a copy of an older lane's target dir (own copy, own path); the build below touches the sources so cargo rebuilds what changed
rsh "[ -d $R/target ] || cp -a ~/intarsia-ab35b/src/target $R/target"
rsh "rm -rf $L/stage && mkdir -p $L/stage"
git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes "$HOSTN" "tar -x -C $L/stage"
rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
date +%T
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli -j 4 2>&1 | grep -E '^(error|warning: unused)|Finished|Compiling intarsia' | tail -20; cp $R/target/release/intarsia $L/bin/intarsia && $L/bin/intarsia --version | head -1"
rsh "df -h ~ | tail -1; du -sh $L; free -g | sed -n 2p"
date +%T
echo "git sha $SHA"
