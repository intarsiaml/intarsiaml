#!/bin/bash
# The PR's binary only (its ids are compared afterwards with tp35f/full/main-*.ids): full stage list once (ids, worker kill, restart, primary kill, SIGTERM and SIGKILL idle), then 4 more worker kills.
cd "$(dirname "$0")/../.."
export SPARK_DIR='~/intarsia-tp35-lost' BIN_MAIN='$HOME/intarsia-tp35-exit/bin/intarsia' BIN_HEAD='$HOME/intarsia-tp35-lost/bin/intarsia'
OUT=$PWD/.scratch/tp35h/full ARMS=head scripts/dsv41-tp2/tp35-compare.sh
for i in 1 2 3 4; do OUT=$PWD/.scratch/tp35h/f1-$i ARMS=head STAGES=f1 scripts/dsv41-tp2/tp35-compare.sh; done
