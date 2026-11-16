#!/usr/bin/env bash
# flipprep: build the CUDA CLI of three revisions on spark2 into ~/intarsia-flipprep/bin (main, p1 = PR 1, flip = PR 2) and copy them to spark1.
# REVS="main=<sha> p1=<sha> flip=<sha>". Seeded from another lane's target dir. Run under:
#   fleetq run --host spark2 --load cpu --mem 34 --expect 60 --label "flipprep: build" -- scripts/flipprep/build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
H=spark2; L='~/intarsia-flipprep'; R="$L/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes $H "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R $L/bin $L/out"
rsh "[ -d $R/target ] || { [ -d ~/intarsia-ab35c-fix/src/target ] && cp -a ~/intarsia-ab35c-fix/src/target $R/target; true; }"
scp -q -o BatchMode=yes "$REPO/scripts/flipprep/memwatch.sh" $H:intarsia-flipprep/memwatch.sh
for spec in ${REVS:?REVS}; do
  name=${spec%%=*}; SHA=$(git -C "$REPO" rev-parse "${spec#*=}"); STAMP="flipprep-$name-${SHA:0:12}"
  echo "== $name $SHA"; date +%T
  rsh "rm -rf $L/stage && mkdir -p $L/stage"
  git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes $H "tar -x -C $L/stage"
  rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
  rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
  rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished|Compiling intarsia' | tail -20; cp $R/target/release/intarsia $L/bin/intarsia-$name && $L/bin/intarsia-$name --version | head -1"
  date +%T
done
rsh "df -h ~ | tail -1; du -sh $L; free -g | sed -n 2p; systemctl is-active earlyoom"
ssh -n -o BatchMode=yes spark1 "mkdir -p ~/intarsia-flipprep/bin ~/intarsia-flipprep/out"
for spec in $REVS; do n=${spec%%=*}; rsh "scp -q -o BatchMode=yes $L/bin/intarsia-$n spark1:intarsia-flipprep/bin/intarsia-$n"; done
ssh -n -o BatchMode=yes spark1 "ls -la ~/intarsia-flipprep/bin"
