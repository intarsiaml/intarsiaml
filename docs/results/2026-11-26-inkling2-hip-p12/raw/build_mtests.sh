#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job: build the intarsia-models test binaries the gates run (--no-run), in this lane's volume.
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-p12"
mkdir -p logs
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-p12 INTARSIA_CONTAINER_NAME=ink-hip-p12-build
trap 'docker rm -f ink-hip-p12-build >/dev/null 2>&1' EXIT
for t in ${MTESTS:-inkling2 inkling2_dspark inkling2_dspark_cut inkling2_verify_softcap inkling2_batch}; do
  deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-models --test "$t" --no-run < /dev/null > logs/build-mtest-$t.log 2>&1
  rc=$?; tail -1 logs/build-mtest-$t.log; echo "models test $t build exit $rc"; [ $rc -ne 0 ] && exit $rc
done
exit 0
