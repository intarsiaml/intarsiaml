#!/usr/bin/env bash
# lane mtp-store: build the CUDA CLI and the `mtp_store_cuda` gate of commit $SHA (default: this worktree's HEAD) on spark1 into
# ~/intarsia-mtp-store (scripts/interleave/build.sh plus the test binary). Run under:
#   fleetq run --host spark1 --load cpu --mem 34 --expect 30 --label "mtp-store: build" -- bash scripts/mtp-store/build.sh
# SEED: a target dir of a recent build on the host (saves the dependencies), copied once. BINNAME: the binary's name under bin/
# (default intarsia; `intarsia-main` for the origin/main arm of the store-off check). NOTEST=1: the CLI only.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L='~/intarsia-mtp-store'; R="$L/src"; H=spark1
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$H" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SHA=$(git -C "$REPO" rev-parse "${SHA:-HEAD}^{commit}"); STAMP="mtp-store-${SHA:0:12}"
rsh "mkdir -p $R $L/bin $L/out"
[[ -n "${SEED:-}" ]] && rsh "test -d $R/target || cp -a $SEED $R/target"
rsh "rm -rf $L/stage && mkdir -p $L/stage"
git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes "$H" "tar -x -C $L/stage"
rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
date +%T
# detached, with a log: the caller polls $L/out/build.log (no tool call waits longer than 4 min)
rsh "cd $R && nohup env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 bash -c 'cargo build --release --locked --features cuda -p intarsia-cli && cp target/release/intarsia ../bin/${BINNAME:-intarsia} && ${NOTEST:+true} cargo test --release --locked -p intarsia-models --test mtp_store_cuda --test prefix_store_cuda --no-run; echo BUILD-EXIT \$?' > $L/out/build.log 2>&1 < /dev/null &"
# wait here (the fleetq slot is held for the build), watching memory
while rsh "pgrep -f 'cargo (build|test) --release --locked' > /dev/null"; do
  a=$(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo")
  echo "$(date +%T) MemAvailable ${a} MB; $(rsh "grep -E 'Compiling intarsia|Finished|^error' $L/out/build.log | tail -1")"
  if [[ "$a" -lt 10240 ]]; then rsh "pkill -9 -f 'cargo (build|test) --release'; pkill -9 nvcc; pkill -9 rustc"; echo "KILLED: MemAvailable ${a} MB"; exit 1; fi
  sleep 20
done
rsh "grep -E '^(error|warning: unused)|Finished|BUILD-EXIT' $L/out/build.log | tail -12; $L/bin/${BINNAME:-intarsia} --version | head -1; df -h ~ | tail -1; du -sh $L; free -g | sed -n 2p"
date +%T
echo "git sha $SHA"
rsh "grep -q 'BUILD-EXIT 0' $L/out/build.log"
