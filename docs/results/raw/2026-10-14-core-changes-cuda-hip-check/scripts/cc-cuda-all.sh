#!/bin/bash
export FLEETQ_JOB=1864
OUT=/tmp/cc-logs; mkdir -p $OUT
/tmp/cc-checks.sh cuda $OUT head kernels optable
for V in base head head-nofusion; do /tmp/cc-checks.sh cuda $OUT $V accshort acclong gen-short gen-4k; done
/tmp/cc-checks.sh cuda $OUT base optable
echo CUDAALLDONE > $OUT/DONE
