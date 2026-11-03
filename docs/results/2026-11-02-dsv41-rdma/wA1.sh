#!/bin/bash
# window A1: RDMA on, greedy smoke + golden (640 prompts), binary of the head
cd /opt/code/intarsiaml/.worktrees/dsv41-rdma
export SPARK_DIR='~/intarsia-dsv41-rdma'
OUT=$PWD/.scratch/wA1; mkdir -p $OUT
OUT=$OUT EARLYOOM_OK="3,2 2,1" STEPS="greedy golden" XENV="INTARSIA_DSV41_RDMA=1" MAX_MIN=14 scripts/dsv41-tp2/run-full.sh
