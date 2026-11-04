#!/bin/bash
# main = the rdma binary, whose crates/ and kernels/ equal merged main 47af18777; head = the fix.
cd "$(dirname "$0")/../.."
export SPARK_DIR='~/intarsia-tp35-exit' BIN_MAIN='$HOME/intarsia-dsv41-rdma/bin/intarsia' BIN_HEAD='$HOME/intarsia-tp35-exit/bin/intarsia'
OUT=$PWD/.scratch/tp35f/full scripts/dsv41-tp2/tp35-compare.sh
for i in 1 2 3; do OUT=$PWD/.scratch/tp35f/f2-$i ARMS=head STAGES=f2 scripts/dsv41-tp2/tp35-compare.sh; done
