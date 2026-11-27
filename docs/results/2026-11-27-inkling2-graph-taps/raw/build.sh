#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (lane ink-plain-tax): build the intarsia-models test target inkling2_dflash2 of the synced
# commit in the lane's own target volume and copy the executable to .scratch/bin/t-<NAME>; CPU=1 also runs the cpu tiny suites.
set -uo pipefail
LANE="$HOME/intarsiaml-ink-plain-tax"; NAME=${1:?name}
cd "$LANE"; mkdir -p logs .scratch/bin
export INTARSIA_REPO=$LANE INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-plain-tax INTARSIA_CONTAINER_NAME=ink-plain-tax-build
trap 'docker rm -f ink-plain-tax-build >/dev/null 2>&1' EXIT
DEV="$LANE/deploy/tatooine/dev.sh"
find "$LANE/crates" "$LANE/kernels" "$LANE/recipes" -type f -exec touch {} +
echo "[build] $NAME commit $(cat .scratch/commit) $(date +%T)"
$DEV cargo test --release -j8 -p intarsia-models --test inkling2_dflash2 --no-run < /dev/null > logs/build-$NAME.log 2>&1
rc=$?; tail -2 logs/build-$NAME.log; echo "[build] test build exit $rc $(date +%T)"; [ $rc -ne 0 ] && exit $rc
exe=$(grep -aoE '\(/?target/release/deps/inkling2_dflash2-[0-9a-f]+\)' logs/build-$NAME.log | tail -1 | tr -d '()')
$DEV cp "/${exe#/}" "/work/.scratch/bin/t-$NAME" < /dev/null && cp .scratch/commit .scratch/bin/t-$NAME.commit && md5sum .scratch/bin/t-$NAME
if [ "${CPU:-0}" = 1 ]; then
  for t in inkling2_dflash2 inkling2_dspark inkling2_dspark_sampled inkling2_dspark_sampled_adapt inkling2_sampled_default; do
    $DEV cargo test --release -j8 -p intarsia-models --test $t -- --test-threads=4 < /dev/null > logs/cpu-$NAME-$t.log 2>&1
    echo "[cpu] $t rc $? | $(grep -aE '^test result' logs/cpu-$NAME-$t.log | tail -1)"
  done
  $DEV cargo test --release -j8 -p intarsia-core --lib < /dev/null > logs/cpu-$NAME-core.log 2>&1
  echo "[cpu] core lib rc $? | $(grep -aE '^test result' logs/cpu-$NAME-core.log | tail -1)"
fi
echo "[build] done $(date +%T)"
