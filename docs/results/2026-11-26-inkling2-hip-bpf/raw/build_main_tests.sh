#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (lane ink-hip-bpf): main's tree (.scratch/main-src = git archive of origin/main; its
# commit in .scratch/main-src/COMMIT) in its own target volume: the cli (hip) -> .scratch/bin/intarsia-<name> (arg 1, default
# main2), then the intarsia-models test binaries for main's baseline of the tiny hip suite.
set -uo pipefail
L=$HOME/intarsiaml-ink-hip-bpf
R=$L/.scratch/main-src
NAME=${1:-main2}
cd "$R"
export INTARSIA_REPO=$R INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-bpf-main INTARSIA_CONTAINER_NAME=ink-hip-bpf-build-main
trap 'docker rm -f ink-hip-bpf-build-main >/dev/null 2>&1' EXIT
mkdir -p "$L/logs" "$L/.scratch/bin"
echo "main-src at $(cat COMMIT 2>/dev/null)"
deploy/tatooine/dev.sh cargo build --release -j8 --features hip -p intarsia-cli < /dev/null > "$L/logs/build-main-cli-$NAME.log" 2>&1
rc=$?; tail -2 "$L/logs/build-main-cli-$NAME.log"; echo "main cli build exit $rc"; [ $rc -ne 0 ] && exit $rc
deploy/tatooine/dev.sh cp /target/release/intarsia /work/intarsia-$NAME < /dev/null && mv "$R/intarsia-$NAME" "$L/.scratch/bin/intarsia-$NAME" && md5sum "$L/.scratch/bin/intarsia-$NAME"
for t in inkling2 inkling2_batch; do
  deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-models --test $t --no-run < /dev/null > "$L/logs/build-main-test-$t.log" 2>&1
  rc=$?; tail -2 "$L/logs/build-main-test-$t.log"; echo "main test $t build exit $rc"; [ $rc -ne 0 ] && exit $rc
done
exit 0
