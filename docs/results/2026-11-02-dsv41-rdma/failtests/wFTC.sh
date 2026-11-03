#!/bin/bash
# failure-test window C: cases c0 d1 d2 d3 d4 d5 d7 d8 d9 d6
cd /opt/code/intarsiaml/.worktrees/dsv41-rdma
export SPARK_DIR='~/intarsia-dsv41-rdma'
OUT=$PWD/.scratch/ftC; mkdir -p $OUT
OUT=$OUT EARLYOOM_OK="3,2 2,1" STEPS="" HOOK=$PWD/scripts/dsv41-tp2/failtest.sh CASES="c0 d1 d2 d3 d4 d5 d7 d8 d9 d6" MAX_MIN=28 scripts/dsv41-tp2/run-full.sh
