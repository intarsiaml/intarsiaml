#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job: build the intarsia binary (hip) at the synced tree (origin/main) and copy it to
# .scratch/bin/intarsia-main (statically linked kernels, self-contained). Lane ink-hip-g2.
set -uo pipefail
LANE="$HOME/intarsiaml-ink-hip-g2"
cd "$LANE"
mkdir -p logs .scratch/bin
export INTARSIA_REPO=$LANE INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-g2 INTARSIA_CONTAINER_NAME=ink-hip-g2-build
trap 'docker rm -f ink-hip-g2-build >/dev/null 2>&1' EXIT
DEV="$LANE/deploy/tatooine/dev.sh"
$DEV cargo build --release -j8 --features hip -p intarsia-cli < /dev/null > logs/build-cli-main.log 2>&1
rc=$?
tail -3 logs/build-cli-main.log
echo "cli build exit $rc"
[ $rc -ne 0 ] && exit $rc
$DEV cp /target/release/intarsia /work/.scratch/bin/intarsia-main < /dev/null && md5sum "$LANE/.scratch/bin/intarsia-main"
