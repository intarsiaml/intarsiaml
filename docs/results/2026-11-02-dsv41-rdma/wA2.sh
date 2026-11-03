#!/bin/bash
# window A2: ABAB, decode by kind (lcat), RDMA 0 / 1 / 0 / 1, spec trace on in all
cd /opt/code/intarsiaml/.worktrees/dsv41-rdma
export SPARK_DIR='~/intarsia-dsv41-rdma'
OUT=$PWD/.scratch/wA2; mkdir -p $OUT
T="INTARSIA_DSV41_SPEC_TRACE=1"
AB="INTARSIA_DSV41_RDMA=0 $T|INTARSIA_DSV41_RDMA=1 $T|INTARSIA_DSV41_RDMA=0 $T|INTARSIA_DSV41_RDMA=1 $T" OUT=$OUT EARLYOOM_OK="3,2 2,1" STEPS="lcat" CAT_TOKENS=256 MAX_MIN=24 scripts/dsv41-tp2/run-full.sh
