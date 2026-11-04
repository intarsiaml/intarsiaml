#!/bin/bash
# Worker killed mid-decode, 8 times on the next_reply fix, the primary not under gdb (the run that hung once was not either).
cd "$(dirname "$0")/../.."
export SPARK_DIR='~/intarsia-tp35-exit' BIN_MAIN='$HOME/intarsia-dsv41-rdma/bin/intarsia' BIN_HEAD='$HOME/intarsia-tp35-exit/bin/intarsia'
for i in 1 2 3 4 5 6 7 8; do
  OUT=$PWD/.scratch/tp35g/f1-$i ARMS=head STAGES=f1 scripts/dsv41-tp2/tp35-compare.sh
done
