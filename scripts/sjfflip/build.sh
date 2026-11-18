#!/usr/bin/env bash
# sjfflip: build the CUDA CLI of a git ref on spark1 into ~/intarsiaml-sjf/bin/intarsia-<tag> (the warm src and target dir of lane sjf,
# so the builds are incremental). Run under: fleetq run --host spark1 --load cpu --mem 40 --expect 45 --label "sjfflip build <tag>" -- scripts/sjfflip/build.sh <tag> <ref>
#   tag main: ref origin/main (the merge base of the branch), tag new: HEAD.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TAG=${1:?tag}; REF=${2:?git ref}
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L='~/intarsiaml-sjf'; R="$L/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes spark1 "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SHA=$(git -C "$REPO" rev-parse "$REF"); STAMP="sjfflip-$TAG-${SHA:0:12}"
rsh "mkdir -p $R $L/bin $L/out"
rsh "rm -rf $L/stage && mkdir -p $L/stage"
git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes spark1 "tar -x -C $L/stage"
rsh "rsync -a --checksum --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
date +%T; rsh "free -g | sed -n 2p"
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished|Compiling intarsia' | tail -20; cp $R/target/release/intarsia $L/bin/intarsia-$TAG && $L/bin/intarsia-$TAG --version | head -1"
rsh "df -h ~ | tail -1; free -g | sed -n 2p"
date +%T; echo "git sha $SHA tag $TAG"
