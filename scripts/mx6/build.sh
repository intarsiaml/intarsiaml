#!/usr/bin/env bash
# mx6: build the CUDA CLI of the mx6 branch on BUILD_HOST (default spark2) into ~/intarsia-mx6/bin, take lane flipprep's origin/main binary as `main`, copy both to the other Spark. REVS="main=<sha> mx6=<sha>". Seeded from another lane's target dir. Run under:
#   fleetq run --host spark2 --load cpu --mem 34 --expect 25 --label "mx6: build" -- env REVS="main=<sha> mx6=<sha>" scripts/mx6/build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
H=${BUILD_HOST:-spark2}; O=$([ "$H" = spark1 ] && echo spark2 || echo spark1); L='~/intarsia-mx6'; R="$L/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes $H "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R $L/bin $L/out"
rsh "[ -d $R/target ] || { d=\$(ls -d ~/intarsia-flipprep/src/target ~/intarsia-stream-burst/src/target 2>/dev/null | head -1); [ -n \"\$d\" ] && cp -a \$d $R/target; true; }"
scp -q -o BatchMode=yes "$REPO/scripts/mx6/memwatch.sh" $H:intarsia-mx6/memwatch.sh
for spec in ${REVS:?REVS}; do
  name=${spec%%=*}; SHA=$(git -C "$REPO" rev-parse "${spec#*=}"); STAMP="mx6-$name-${SHA:0:12}"
  echo "== $name $SHA"; date +%T
  rsh "rm -rf $L/stage && mkdir -p $L/stage"
  git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes $H "tar -x -C $L/stage"
  rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
  rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
  rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished|Compiling intarsia' | tail -20; cp $R/target/release/intarsia $L/bin/intarsia-$name && $L/bin/intarsia-$name --version | head -1"
  date +%T
done
# the `main` reference binary: when REVS has no `main=`, take lane flipprep's origin/main CUDA build (base c5ce87bc8; origin/main has changed only
# crates/intarsia-core/src/hostprof.rs, a test, since: `git diff c5ce87bc8 origin/main -- crates kernels recipes Cargo.toml Cargo.lock`)
rsh "[ -f $L/bin/intarsia-main ] || cp -p ~/intarsia-flipprep/bin/intarsia-main $L/bin/intarsia-main; ls -la $L/bin; sha256sum $L/bin/intarsia-main | cut -c1-16"
rsh "df -h ~ | tail -1; du -sh $L; free -g | sed -n 2p; systemctl is-active earlyoom"
ssh -n -o BatchMode=yes $O "mkdir -p ~/intarsia-mx6/bin ~/intarsia-mx6/out"
for n in main mx6; do rsh "scp -q -o BatchMode=yes $L/bin/intarsia-$n $O:intarsia-mx6/bin/intarsia-$n"; done
scp -q -o BatchMode=yes "$REPO/scripts/mx6/memwatch.sh" $O:intarsia-mx6/memwatch.sh
ssh -n -o BatchMode=yes $O "ls -la ~/intarsia-mx6/bin"
