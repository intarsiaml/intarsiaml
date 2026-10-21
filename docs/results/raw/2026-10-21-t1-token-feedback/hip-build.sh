#!/usr/bin/env bash
# ON tatooine, non-exclusive fleetq slot (GPUs untouched, radiance up): build base (origin/main) and dev (T1) in the
# HIP dev image, each in its own target volume; dev also runs the CPU-side core/models tests (GPUs hidden).
# usage: bash hip-build.sh   (trees in ~/t1/base and ~/t1/dev)
set -uo pipefail
OUT=$HOME/t1/out; mkdir -p $OUT
for T in base dev; do
  export INTARSIA_REPO=$HOME/t1/$T INTARSIA_TARGET_VOL=intarsia-cargo-target-t1$T
  D=$HOME/t1/$T/deploy/tatooine/dev.sh
  echo "== $T build $(date '+%F %T')"
  $D cargo build --release --jobs 8 -p intarsia-cli --features hip > $OUT/build-$T.log 2>&1; echo "$T build exit $?"
done
export INTARSIA_REPO=$HOME/t1/dev INTARSIA_TARGET_VOL=intarsia-cargo-target-t1dev HIP_VISIBLE_DEVICES=-1
echo "== dev cpu tests $(date '+%F %T')"
$HOME/t1/dev/deploy/tatooine/dev.sh cargo test --release --jobs 8 -p intarsia-core -p intarsia-models > $OUT/test-dev-cpu.log 2>&1; echo "dev cpu tests exit $?"
grep -h "test result" $OUT/test-dev-cpu.log | awk '{p+=$4; f+=$6} END {print "total passed", p, "failed", f}'
echo HIP_BUILD_DONE $(date '+%F %T')
