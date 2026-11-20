#!/usr/bin/env bash
# kvflip: sync this worktree's HEAD to spark1 ~/intarsiaml-kvflip/src, build the CUDA CLI (and with TESTS="kv_once attention ..." the CUDA
# backend's test binaries, copied to ~/intarsiaml-kvflip/bin/<name>-test), then copy the CLI to spark2 ~/intarsiaml-kvflip/bin/<BIN_NAME>.
#   [TESTS="kv_once attention verify_attn_rows"] [BIN_NAME=intarsia] FLEETQ_OWNER=claude-kvflip fleetq run --host spark1 --load cpu --mem 40 --expect 40 --label "kvflip build" -- bash scripts/kvflip/build.sh
# Watchdog: polls MemAvailable every 20 s; kills the build's process group below 15 GB. Remove ~/intarsiaml-kvflip on both Sparks when the lane is done.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L='~/intarsiaml-kvflip'; R="$L/src"; BH=spark1; PEER=spark2; BN=${BIN_NAME:-intarsia}
rsh() { ssh -n -o BatchMode=yes "$BH" "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SHA=$(git -C "$REPO" rev-parse HEAD); STAMP="kvflip-${SHA:0:12}"
rsh "mkdir -p $R $L/bin $L/out && rm -rf $L/stage && mkdir -p $L/stage"
rsh "test -d $R/target || { test -d ~/intarsia-acc35b/src/target && cp -a ~/intarsia-acc35b/src/target $R/target; true; }"
git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes "$BH" "tar -x -C $L/stage"
rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
echo "build start $(date +%T) sha $SHA"
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
watched "env INTARSIA_BUILD_STAMP='$STAMP' cargo build --release --locked --features cuda -p intarsia-cli"
rsh "cp $R/target/release/intarsia $L/bin/$BN && $L/bin/$BN --version | head -1"
rsh "ssh -o BatchMode=yes $PEER 'mkdir -p $L/bin $L/out' && scp -q $L/bin/$BN $PEER:$L/bin/$BN.new && ssh $PEER 'mv $L/bin/$BN.new $L/bin/$BN && $L/bin/$BN --version | head -1'"
rsh "df -h ~ | tail -1; free -g | sed -n 2p"
echo "build end $(date +%T) stamp $STAMP"
