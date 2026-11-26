#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job: build origin/main (the tree synced to .scratch/main-tree) in the lane's volume under its
# own target dir (/target/mt, so the branch's incremental cache stays), copy it to .scratch/bin/intarsia-<name>.
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-batch"
mkdir -p logs .scratch/bin
NAME=${1:?name}
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-batch INTARSIA_CONTAINER_NAME=ink-hip-batch-build
trap 'docker rm -f ink-hip-batch-build >/dev/null 2>&1' EXIT
deploy/tatooine/dev.sh bash -c 'cd /work/.scratch/main-tree && CARGO_TARGET_DIR=/target/mt cargo build --release -j8 --features hip -p intarsia-cli && cp /target/mt/release/intarsia /work/.scratch/bin/intarsia-'"$NAME" < /dev/null > logs/build-main-$NAME.log 2>&1
rc=$?
tail -3 logs/build-main-$NAME.log
echo "main build exit $rc"
[ $rc -eq 0 ] && md5sum .scratch/bin/intarsia-$NAME
exit $rc
