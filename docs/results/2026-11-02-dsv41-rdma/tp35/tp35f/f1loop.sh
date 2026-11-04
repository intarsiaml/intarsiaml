#!/bin/bash
# Worker killed mid-decode, 5 times per binary, the primary under gdb: stuck requests end as curl rc 52 with a stack in the serve log.
cd "$(dirname "$0")/../.."
export SPARK_DIR='~/intarsia-tp35-exit' BIN_MAIN='$HOME/intarsia-dsv41-rdma/bin/intarsia' BIN_HEAD='$HOME/intarsia-tp35-exit/bin/intarsia'
for i in 1 2 3 4 5; do
  for arm in main head; do
    OUT=$PWD/.scratch/tp35f/f1-$arm-$i ARMS=$arm STAGES=f1 SERVE_GDB=1 scripts/dsv41-tp2/tp35-compare.sh
  done
done
