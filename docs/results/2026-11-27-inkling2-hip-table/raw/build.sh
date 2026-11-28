#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job: build the intarsia binary (hip) in the lane's volume; with a name, copy it to
# .scratch/bin/intarsia-<name> (the binary links the kernels statically, so the copy is self-contained); then build the
# intarsia-models test targets in $MTESTS (--no-run; default none).
# (After lane ink-hip-spec's build.sh: synced files older than a build of another tree in the seeded volume would count as fresh.)
set -uo pipefail
LANE="$HOME/intarsiaml-ink-hip-table"
NAME=${1:-}
cd "$LANE"
mkdir -p logs .scratch/bin
export INTARSIA_REPO=$LANE INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-table INTARSIA_CONTAINER_NAME=ink-hip-table-build
trap 'docker rm -f ink-hip-table-build >/dev/null 2>&1' EXIT
DEV="$LANE/deploy/tatooine/dev.sh"
find "$LANE/crates" "$LANE/kernels" "$LANE/recipes" -type f -exec touch {} +
$DEV cargo build --release -j8 --features hip -p intarsia-cli < /dev/null > logs/build-cli-$NAME.log 2>&1
rc=$?
tail -3 logs/build-cli-$NAME.log
echo "cli build exit $rc"
[ $rc -ne 0 ] && exit $rc
if [ -n "$NAME" ]; then
  $DEV cp /target/release/intarsia /work/.scratch/bin/intarsia-$NAME < /dev/null && md5sum "$LANE/.scratch/bin/intarsia-$NAME"
fi
for t in ${MTESTS-}; do
  $DEV cargo test --release -j8 -p intarsia-models --test "$t" --no-run < /dev/null > logs/build-mtest-$t.log 2>&1
  rc=$?; tail -2 logs/build-mtest-$t.log; echo "models test $t build exit $rc"; [ $rc -ne 0 ] && exit $rc
  # (lane ink-sampled-adapt's build.sh) the test executable, copied to .scratch/bin/t-<NAME>-<test>
  exe=$(grep -aoE '\(/?target/release/deps/[A-Za-z0-9_]+-[0-9a-f]+\)' logs/build-mtest-$t.log | tail -1 | tr -d '()')
  [ -n "$exe" ] && [ -n "$NAME" ] && $DEV cp "/${exe#/}" "/work/.scratch/bin/t-$NAME-$t" < /dev/null && echo "  -> .scratch/bin/t-$NAME-$t"
done
exit 0
