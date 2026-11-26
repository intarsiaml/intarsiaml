#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job: build the intarsia binary (hip) in the lane's volume; with a name, copy it to
# .scratch/bin/intarsia-<name> (the binary links the kernels statically, so the copy is self-contained).
#   build.sh NAME [SUBDIR]   SUBDIR = source tree under ~/intarsiaml-ink-hip-pfflip (default the branch itself; main: .scratch/main-src)
# Env TESTS="ink_moe ink_attn": test targets of intarsia-backend-hip to build too (--no-run). Env LIBTESTS=1: also run the hip
# crate's lib tests (no GPU needed) and the models recipe tests.
set -uo pipefail
LANE="$HOME/intarsiaml-ink-hip-pfflip"
NAME=${1:-}; SUB=${2:-}
cd "$LANE"
mkdir -p logs .scratch/bin
export INTARSIA_REPO=$LANE${SUB:+/$SUB} INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-pfflip INTARSIA_CONTAINER_NAME=ink-hip-pfflip-build
trap 'docker rm -f ink-hip-pfflip-build >/dev/null 2>&1' EXIT
DEV="$LANE/deploy/tatooine/dev.sh"
# cargo compares source mtimes with the last build's: a tree synced from the Mac whose files are older than a build of another tree
# in the same volume would count as fresh. Touch the sources so this tree always rebuilds.
find "$INTARSIA_REPO/crates" "$INTARSIA_REPO/kernels" "$INTARSIA_REPO/recipes" -type f -exec touch {} +
mkdir -p "$INTARSIA_REPO/.scratch/bin"
$DEV cargo build --release -j8 --features hip -p intarsia-cli < /dev/null > logs/build-cli-$NAME.log 2>&1
rc=$?
tail -3 logs/build-cli-$NAME.log
echo "cli build exit $rc"
[ $rc -ne 0 ] && exit $rc
if [ -n "$NAME" ]; then
  $DEV cp /target/release/intarsia /work/.scratch/bin/intarsia-$NAME < /dev/null && md5sum "$INTARSIA_REPO/.scratch/bin/intarsia-$NAME"
  [ -n "$SUB" ] && cp "$INTARSIA_REPO/.scratch/bin/intarsia-$NAME" "$LANE/.scratch/bin/intarsia-$NAME" 2>/dev/null
fi
for t in ${TESTS:-}; do
  $DEV cargo test --release -j8 -p intarsia-backend-hip --test "$t" --no-run < /dev/null > logs/build-test-$t.log 2>&1
  rc=$?; tail -2 logs/build-test-$t.log; echo "test $t build exit $rc"; [ $rc -ne 0 ] && exit $rc
done
if [ -n "${LIBTESTS:-}" ]; then
  $DEV cargo test --release -j8 -p intarsia-backend-hip --lib < /dev/null > logs/test-hip-lib.log 2>&1
  echo "hip lib tests exit $?"; grep -aE "^test result|FAILED|panicked|failed" logs/test-hip-lib.log | head
  $DEV cargo test --release -j8 -p intarsia-backend-hip --test ink_moe --test ink_attn --no-run < /dev/null > logs/build-test-ink.log 2>&1
  echo "hip ink_moe / ink_attn test build exit $?"; tail -2 logs/build-test-ink.log
fi
exit 0
