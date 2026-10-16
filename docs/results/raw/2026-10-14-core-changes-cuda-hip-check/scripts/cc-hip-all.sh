#!/bin/bash
export FLEETQ_JOB=1865
OUT=/tmp/cc-logs-hip; mkdir -p $OUT
until grep -q "^EXIT" /tmp/hip-head-build.log && grep -q "^EXIT" /tmp/hip-base-build.log; do sleep 3; done
/tmp/cc-checks.sh hip $OUT head kernels optable
( export INTARSIA_REPO=/tmp/intarsia-hip-head INTARSIA_TARGET_VOL=ccheck-target-head HIP_VISIBLE_DEVICES=1
  /tmp/intarsia-hip-head/deploy/tatooine/dev.sh cargo test --release -p intarsia-backend-hip > $OUT/cargo-test-hip.log 2>&1; echo "EXIT $?" >> $OUT/cargo-test-hip.log )
for V in base head head-nofusion; do /tmp/cc-checks.sh hip $OUT $V accshort acclong gen-short gen-4k; done
/tmp/cc-checks.sh hip $OUT base optable
echo HIPALLDONE > $OUT/DONE
