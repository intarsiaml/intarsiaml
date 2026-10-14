#!/bin/bash
# usage: cc-hipbuild.sh base|head
T=$1
export FLEETQ_JOB=1865 INTARSIA_REPO=/tmp/intarsia-hip-$T INTARSIA_TARGET_VOL=ccheck-target-$T
/tmp/intarsia-hip-head/deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip > /tmp/hip-$T-build.log 2>&1; echo "EXIT $?" >> /tmp/hip-$T-build.log
