#!/usr/bin/env bash
# cvat: sync this worktree (or REF=<git ref> with OUT=<binary name>) to spark1 ~/intarsiaml-cvatflip/src and build the CUDA CLI. Run under:
#   fleetq run --host spark1 --load cpu --mem 55 --expect 45 --label "cvatflip build" -- bash scripts/cvatflip/build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L='~/intarsiaml-cvatflip'; R="$L/src"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes spark1 "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
# the worktree's tracked files plus its uncommitted changes (the lane edits before it commits)
rsh "mkdir -p $R $L/bin $L/out && rm -rf $L/stage && mkdir -p $L/stage"
# REF=<git ref>: build that commit's tree instead (the baseline binary, OUT=intarsia-main)
if [ -n "${REF:-}" ]; then SRC=(git -C "$REPO" archive --format=tar "$REF" -- . ':(exclude)docs/results'); else SRC=(sh -c 'cd "$1" && COPYFILE_DISABLE=1 git ls-files -z --cached --others --exclude-standard | grep -zv "^docs/results/\|^target-endor/\|^\.worktrees/" | COPYFILE_DISABLE=1 tar --no-xattrs --null -T - -cf -' _ "$REPO"); fi
OUT=${OUT:-intarsia}
( "${SRC[@]}" ) | ssh -o BatchMode=yes spark1 "tar --warning=no-unknown-keyword -x -C $L/stage"
rsh "find $L/stage -name '._*' -delete; rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.toml' \) -print0 | xargs -0 touch"
[ -n "${CLEAN_CUDA:-}" ] && rsh "rm -rf $R/target/release/build/intarsia-backend-cuda-* $R/target/release/.fingerprint/intarsia-backend-cuda-*"
STAMP="cvat-$(git -C "$REPO" rev-parse --short=12 ${REF:-HEAD})$([ -n "${REF:-}" ] || echo -dirty)"
scp -q -o BatchMode=yes "$(dirname "${BASH_SOURCE[0]}")/wd.sh" spark1:intarsiaml-cvatflip/wd.sh
rsh "nohup bash $L/wd.sh > $L/out/wd.log 2>&1 < /dev/null & echo \$! > $L/out/wd.pid"
trap 'rsh "kill \$(cat $L/out/wd.pid)" || true' EXIT
date +%T
if [ -n "${TESTS:-}" ]; then
  # TESTS="verify_attn_rows": build the CUDA backend test binary and copy it to $L/bin/<name>-test
  rsh "cd $R && env CARGO_BUILD_JOBS=3 MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1 cargo test --release --locked -p intarsia-backend-cuda --test $TESTS --no-run > $L/out/build-test.log 2>&1; tail -5 $L/out/build-test.log; f=\$(ls -t $R/target/release/deps/$TESTS-* | grep -v '\\.d\$' | head -1); cp \$f $L/bin/$TESTS-test && echo copied \$f"
  rsh "df -h ~ | tail -1; free -g | sed -n 2p"; date +%T; exit 0
fi
rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=3 MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli > $L/out/build.log 2>&1; grep -E '^(error|warning: unused)|Finished|Compiling intarsia' $L/out/build.log | tail -20; grep -B2 -A25 'failed to run custom build' $L/out/build.log | head -60; cp $R/target/release/intarsia $L/bin/$OUT && $L/bin/$OUT --version | head -1"
rsh "df -h ~ | tail -1; du -sh $L; free -g | sed -n 2p"
date +%T
