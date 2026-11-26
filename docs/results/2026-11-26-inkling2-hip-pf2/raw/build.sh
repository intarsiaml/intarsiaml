#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job: build the intarsia binary (hip) in the lane's volume; with a name, copy it to
# .scratch/bin/intarsia-<name> (the binary links the kernels statically, so the copy is self-contained). Extra args: test targets
# of intarsia-backend-hip to build too (--no-run).
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-pf2"
mkdir -p logs .scratch/bin
NAME=${1:-}; shift || true
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-pf2 INTARSIA_CONTAINER_NAME=ink-hip-pf2-build
trap 'docker rm -f ink-hip-pf2-build >/dev/null 2>&1' EXIT
deploy/tatooine/dev.sh cargo build --release -j8 --features hip -p intarsia-cli < /dev/null > logs/build-cli-$NAME.log 2>&1
rc=$?
tail -3 logs/build-cli-$NAME.log
echo "cli build exit $rc"
[ $rc -ne 0 ] && exit $rc
if [ -n "$NAME" ]; then
  deploy/tatooine/dev.sh cp /target/release/intarsia /work/.scratch/bin/intarsia-$NAME < /dev/null && md5sum .scratch/bin/intarsia-$NAME
fi
for t in "$@"; do
  deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-backend-hip --test "$t" --no-run < /dev/null > logs/build-test-$t.log 2>&1
  rc=$?; tail -2 logs/build-test-$t.log; echo "test $t build exit $rc"; [ $rc -ne 0 ] && exit $rc
done
exit 0
