#!/usr/bin/env bash
# itl2s: build the CUDA CLI of REV (default HEAD) on spark2 into ~/intarsia-itl2s/bin/intarsia-<NAME>. NAME=itl2s REV=HEAD
# Run under: fleetq run --host spark2 --load cpu --mem 34 --expect 45 --label "itl2s: build" -- env NAME=itl2s REV=HEAD scripts/itl2s/build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
H=spark2; L='~/intarsia-itl2s'; R="$L/src"; NAME=${NAME:-itl2s}; REV=${REV:-HEAD}
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes $H "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R $L/bin $L/out $L/scripts"
rsh "[ -d $R/target ] || { [ -d ~/intarsia-soak/src/target ] && cp -a ~/intarsia-soak/src/target $R/target; true; }"
scp -q -o BatchMode=yes "$REPO/scripts/itl2s/memwatch.sh" $H:intarsia-itl2s/memwatch.sh
SHA=$(git -C "$REPO" rev-parse "$REV"); STAMP="itl2s-$NAME-${SHA:0:12}"
echo "== $NAME $SHA"; date +%T
rsh "rm -rf $L/stage && mkdir -p $L/stage"
git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes $H "tar -x -C $L/stage"
rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rsync -a $L/stage/scripts/ $L/scripts/ && rm -rf $L/stage"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished|Compiling intarsia' | tail -20; cp $R/target/release/intarsia $L/bin/intarsia-$NAME && $L/bin/intarsia-$NAME --version | head -1"
date +%T
rsh "df -h ~ | tail -1; du -sh $L; free -g | sed -n 2p; systemctl is-active earlyoom"
