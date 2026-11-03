#!/bin/bash
# failure-test window A: cases c0 a1 a2 a3 a4
cd /opt/code/intarsiaml/.worktrees/dsv41-rdma
export SPARK_DIR='~/intarsia-dsv41-rdma'
OUT=$PWD/.scratch/ftA; mkdir -p $OUT
OUT=$OUT EARLYOOM_OK="3,2 2,1" STEPS="" HOOK=$PWD/scripts/dsv41-tp2/failtest.sh CASES="c0 a1 a2 a3 a4" MAX_MIN=26 scripts/dsv41-tp2/run-full.sh
