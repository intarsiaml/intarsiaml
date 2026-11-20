#!/usr/bin/env bash
# kvonce: sync this worktree's HEAD to spark2 ~/intarsiaml-kvonce/src and build the CUDA CLI (or, with TESTS="kv_once ...", the CUDA
# backend's test binaries, copied to ~/intarsiaml-kvonce/bin/<name>-test), with a free-memory watchdog.
#   [TESTS="kv_once ..." [CLI=1]] FLEETQ_OWNER=claude-kvonce fleetq run --host spark2 --load cpu --mem 55 --expect 25 --label "kvonce build" -- bash scripts/kvonce/build.sh
# Watchdog: polls MemAvailable every 20 s; kills the build's process group (by PGID) below 15 GB. Remove ~/intarsiaml-kvonce when the lane is done.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L='~/intarsiaml-kvonce'; R="$L/src"; BH=spark2
rsh() { ssh -n -o BatchMode=yes "$BH" "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SHA=$(git -C "$REPO" rev-parse HEAD); STAMP="kvonce-${SHA:0:12}"
rsh "mkdir -p $R $L/bin $L/out && rm -rf $L/stage && mkdir -p $L/stage"
git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes "$BH" "tar -x -C $L/stage"
rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
echo "build start $(date +%T) sha $SHA"
# run one cargo command on the box under the memory watchdog
watched() {
  local PGID; PGID=$(rsh "cd $R; CARGO_BUILD_JOBS=3 MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1 setsid nohup $1 > $L/out/build.log 2>&1 < /dev/null & echo \$!")
  echo "$(date +%T) cargo pgid $PGID: $1"
  while rsh "kill -0 $PGID" 2>/dev/null; do
    a=$(rsh "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo")
    echo "$(date +%T) avail ${a} GB"
    if [[ -n $a && $a -lt 15 ]]; then echo "WATCHDOG: available $a GB < 15, killing build pgid $PGID"; rsh "kill -TERM -$PGID"; sleep 5; rsh "kill -KILL -$PGID 2>/dev/null"; exit 3; fi
    sleep 20
  done
  rsh "grep -E '^(error|warning: unused)|Finished' $L/out/build.log | tail -8; grep -B3 -A20 '^error' $L/out/build.log | head -60"
}
if [[ -n ${TESTS:-} ]]; then
  TARGS=""; for t in $TESTS; do TARGS="$TARGS --test $t"; done
  watched "cargo test --release --locked -p intarsia-backend-cuda $TARGS --no-run"
  for t in $TESTS; do rsh "f=\$(ls -t $R/target/release/deps/$t-* | grep -v '\\.d\$' | head -1); cp \$f $L/bin/$t-test && echo copied \$f"; done
fi
if [[ -z ${TESTS:-} || -n ${CLI:-} ]]; then
  watched "env INTARSIA_BUILD_STAMP='$STAMP' cargo build --release --locked --features cuda -p intarsia-cli"
  rsh "cp $R/target/release/intarsia $L/bin/${BIN_NAME:-intarsia} && $L/bin/${BIN_NAME:-intarsia} --version | head -1"
fi
rsh "df -h ~ | tail -1; free -g | sed -n 2p"
echo "build end $(date +%T) stamp $STAMP"
