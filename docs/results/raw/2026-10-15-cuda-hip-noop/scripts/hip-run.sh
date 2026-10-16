#!/bin/bash
# usage: hip-run.sh <A|B>   (on tatooine inside fleetq job 2003; builds + CPU-side tests only, GPUs hidden)
T=$1
export FLEETQ_JOB=2003 INTARSIA_REPO=/tmp/cnc/$T INTARSIA_TARGET_VOL=cnc-target-$T
D=/tmp/cnc/$T/deploy/tatooine/dev.sh
OUT=/tmp/cnc/out-$T; mkdir -p $OUT
J="--jobs 8"
cd /tmp/cnc/$T
$D bash -c 'echo head: $(cat COMMIT); rustc --version; hipcc --version | head -3' > $OUT/versions.txt 2>&1
$D cargo build --release $J -p intarsia-cli --features hip > $OUT/build.log 2>&1; echo "EXIT $?" >> $OUT/build.log
# GPU tests need the cards: compile only
$D cargo test --release $J -p intarsia-backend-hip --no-run > $OUT/hip-test-build.log 2>&1; echo "EXIT $?" >> $OUT/hip-test-build.log
# CPU-side tests with both GPUs hidden from HIP (HIP_VISIBLE_DEVICES=-1): radiance has the cards
export HIP_VISIBLE_DEVICES=-1
for c in intarsia-core intarsia-models; do
  $D cargo test --release $J -p $c > $OUT/test-$c.log 2>&1; echo "EXIT $?" >> $OUT/test-$c.log
done
$D cargo test --release $J -p intarsia-cli --features hip > $OUT/test-intarsia-cli.log 2>&1; echo "EXIT $?" >> $OUT/test-intarsia-cli.log
$D cargo test --release $J -p intarsia-backend-hip > $OUT/test-intarsia-backend-hip-nogpu.log 2>&1; echo "EXIT $?" >> $OUT/test-intarsia-backend-hip-nogpu.log
echo ALLDONE > $OUT/DONE
