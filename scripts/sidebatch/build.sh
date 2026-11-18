#!/usr/bin/env bash
# sidebatch: sync this worktree (tracked + uncommitted files) to spark2 ~/intarsiaml-sidebatch/src and build the CUDA CLI as bin/intarsia. Run under:
#   fleetq run --host spark2 --load cpu --mem 55 --expect 60 --label "sidebatch: build" -- bash scripts/sidebatch/build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L='~/intarsiaml-sidebatch'; R="$L/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes spark2 "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R $L/bin $L/out && rm -rf $L/stage && mkdir -p $L/stage"
( sh -c 'cd "$1" && COPYFILE_DISABLE=1 git ls-files -z --cached --others --exclude-standard | COPYFILE_DISABLE=1 tar --no-xattrs --null -T - -cf -' _ "$REPO" ) | ssh -o BatchMode=yes spark2 "tar --warning=no-unknown-keyword -x -C $L/stage"
rsh "find $L/stage -name '._*' -delete; rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.toml' \) -print0 | xargs -0 touch"
STAMP="sidebatch-$(git -C "$REPO" rev-parse --short=12 HEAD)-counter"
scp -q -o BatchMode=yes "$(dirname "${BASH_SOURCE[0]}")/wd.sh" spark2:intarsiaml-sidebatch/wd.sh
rsh "nohup bash $L/wd.sh > $L/out/wd.log 2>&1 < /dev/null & echo \$! > $L/out/wd.pid"
trap 'rsh "kill \$(cat $L/out/wd.pid)" || true' EXIT
date +%T
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=3 MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli > $L/out/build.log 2>&1; grep -E '^(error|warning: unused)|Finished|Compiling intarsia' $L/out/build.log | tail -20; grep -B2 -A25 'failed to run custom build' $L/out/build.log | head -60; cp $R/target/release/intarsia $L/bin/intarsia && $L/bin/intarsia --version | head -1"
rsh "df -h ~ | tail -1; du -sh $L; free -g | sed -n 2p"
date +%T
