#!/usr/bin/env bash
# think2: sync this worktree to spark2 ~/intarsiaml-think2/src and build the CUDA CLI, with a free-memory watchdog.
#   FLEETQ_OWNER=claude-overnight-think2 fleetq run --host spark2 --load cpu --mem 55 --expect 25 --label "think2 build" -- bash scripts/think2/build.sh
# Watchdog: polls MemAvailable every 20 s; kills the build's process group (by PGID) below 15 GB. Remove ~/intarsiaml-think2 when the lane is done.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L='~/intarsiaml-think2'; R="$L/src"; BH=spark2
rsh() { ssh -n -o BatchMode=yes "$BH" "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SHA=$(git -C "$REPO" rev-parse 5d78422ae); STAMP="think2-${SHA:0:12}"
rsh "mkdir -p $R $L/bin $L/out && rm -rf $L/stage && mkdir -p $L/stage"
git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes "$BH" "tar -x -C $L/stage"
rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
echo "build start $(date +%T) sha $SHA"
PGID=$(rsh "cd $R; INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 setsid nohup cargo build --release --locked --features cuda -p intarsia-cli > $L/out/build.log 2>&1 < /dev/null & echo \$!")
echo "cargo pgid $PGID"
while rsh "kill -0 $PGID" 2>/dev/null; do
  a=$(rsh "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo")
  echo "$(date +%T) avail ${a} GB"
  if [[ -n $a && $a -lt 15 ]]; then echo "WATCHDOG: available $a GB < 15, killing build pgid $PGID"; rsh "kill -TERM -$PGID"; sleep 5; rsh "kill -KILL -$PGID 2>/dev/null"; exit 3; fi
  sleep 20
done
rsh "tail -5 $L/out/build.log"
rsh "cp $R/target/release/intarsia $L/bin/intarsia && $L/bin/intarsia --version | head -1; df -h ~ | tail -1; free -g | sed -n 2p"
echo "build end $(date +%T) stamp $STAMP"
