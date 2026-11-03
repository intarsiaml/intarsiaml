#!/bin/bash
# failure-test window B: cases c0 b1 b2 c1 c2 c3
cd /opt/code/intarsiaml/.worktrees/dsv41-rdma
export SPARK_DIR='~/intarsia-dsv41-rdma'
OUT=$PWD/.scratch/ftB; mkdir -p $OUT
OUT=$OUT EARLYOOM_OK="3,2 2,1" STEPS="" HOOK=$PWD/scripts/dsv41-tp2/failtest.sh CASES="c0 e1 e2 c1 c2 c3" MAX_MIN=28 scripts/dsv41-tp2/run-full.sh
