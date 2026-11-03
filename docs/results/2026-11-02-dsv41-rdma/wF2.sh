#!/bin/bash
# final head 2dd8dc5bd, shipped defaults, no env overrides: golden 640 (RDMA default-on) + lcat
cd /opt/code/intarsiaml/.worktrees/dsv41-rdma
export SPARK_DIR='~/intarsia-dsv41-rdma'
OUT=$PWD/.scratch/wF2; mkdir -p $OUT
OUT=$OUT EARLYOOM_OK="3,2 2,1" STEPS="golden lcat" CAT_TOKENS=256 MAX_MIN=15 scripts/dsv41-tp2/run-full.sh
