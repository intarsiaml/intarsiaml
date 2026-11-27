#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (lane ink-sampled-adapt): build the intarsia CLI (hip) in the lane's volume and copy it to
# .scratch/bin/intarsia-<NAME>; MTESTS="a b": intarsia-models test binaries too (--no-run), copied to .scratch/bin/t-<NAME>-<test>.
#   build.sh NAME [SUBDIR]   SUBDIR = source tree under the lane folder (default the branch itself; main: .scratch/main-src)
set -uo pipefail
LANE="$HOME/intarsiaml-ink-sampled-adapt"
NAME=${1:?name}; SUB=${2:-}
cd "$LANE"
mkdir -p logs .scratch/bin
export INTARSIA_REPO=$LANE${SUB:+/$SUB} INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-sampled-adapt INTARSIA_CONTAINER_NAME=ink-adapt-build
trap 'docker rm -f ink-adapt-build >/dev/null 2>&1' EXIT
DEV="$LANE/deploy/tatooine/dev.sh"
# a synced tree whose files are older than a build of another tree in the same volume would count as fresh: touch
find "$INTARSIA_REPO/crates" "$INTARSIA_REPO/kernels" "$INTARSIA_REPO/recipes" -type f -exec touch {} +
mkdir -p "$INTARSIA_REPO/.scratch/bin"
$DEV cargo build --release -j8 --features hip -p intarsia-cli < /dev/null > logs/build-cli-$NAME.log 2>&1
rc=$?
tail -2 logs/build-cli-$NAME.log
echo "cli build $NAME exit $rc"
[ $rc -ne 0 ] && exit $rc
$DEV cp /target/release/intarsia /work/.scratch/bin/intarsia-$NAME < /dev/null && md5sum "$INTARSIA_REPO/.scratch/bin/intarsia-$NAME"
[ -n "$SUB" ] && cp "$INTARSIA_REPO/.scratch/bin/intarsia-$NAME" "$LANE/.scratch/bin/intarsia-$NAME"
for t in ${MTESTS:-}; do
  $DEV cargo test --release -j8 -p intarsia-models --test "$t" --no-run < /dev/null > logs/build-mtest-$t.log 2>&1
  rc=$?; tail -1 logs/build-mtest-$t.log; echo "models test $t build exit $rc"; [ $rc -ne 0 ] && exit $rc
  exe=$(grep -aoE '\(/?target/release/deps/[A-Za-z0-9_]+-[0-9a-f]+\)' logs/build-mtest-$t.log | tail -1 | tr -d '()')
  [ -n "$exe" ] && $DEV cp "/${exe#/}" "/work/.scratch/bin/t-$NAME-$t" < /dev/null && echo "  -> .scratch/bin/t-$NAME-$t"
done
exit 0
