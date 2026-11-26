#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job: build main's cli and its tiny `inkling2` test from the main source tree the Mac
# unpacked at .scratch/main-src (git archive of origin/main), in the lane's volume under /target/main (its own target dir, so the
# branch's artifacts are not touched). Outputs: .scratch/bin/intarsia-<name>, .scratch/bin/inkling2-test-<name>.
set -uo pipefail
LANE=$HOME/intarsiaml-ink-hip-p12
NAME=${1:?name}
SRC=$LANE/.scratch/main-src
mkdir -p "$LANE/.scratch/bin" "$LANE/logs" "$SRC/.scratch"
export INTARSIA_REPO=$SRC INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-p12 INTARSIA_CONTAINER_NAME=ink-hip-p12-build-main
trap 'docker rm -f ink-hip-p12-build-main >/dev/null 2>&1' EXIT
DEV="$LANE/deploy/tatooine/dev.sh"
$DEV cargo build --release -j8 --features hip -p intarsia-cli --target-dir /target/main < /dev/null > "$LANE/logs/build-main-$NAME.log" 2>&1
rc=$?; tail -2 "$LANE/logs/build-main-$NAME.log"; echo "main cli build exit $rc"; [ $rc -ne 0 ] && exit $rc
$DEV cp /target/main/release/intarsia /work/.scratch/intarsia < /dev/null && cp "$SRC/.scratch/intarsia" "$LANE/.scratch/bin/intarsia-$NAME" && md5sum "$LANE/.scratch/bin/intarsia-$NAME"
$DEV cargo test --release -j8 -p intarsia-models --test inkling2 --no-run --target-dir /target/main < /dev/null > "$LANE/logs/build-main-test-$NAME.log" 2>&1
rc=$?; tail -1 "$LANE/logs/build-main-test-$NAME.log"; echo "main inkling2 test build exit $rc"; [ $rc -ne 0 ] && exit $rc
exe=$(grep -aoE '/target/main/release/deps/inkling2-[0-9a-f]+' "$LANE/logs/build-main-test-$NAME.log" | tail -1)
$DEV cp "$exe" /work/.scratch/inkling2-test < /dev/null && cp "$SRC/.scratch/inkling2-test" "$LANE/.scratch/bin/inkling2-test-$NAME" && md5sum "$LANE/.scratch/bin/inkling2-test-$NAME"
