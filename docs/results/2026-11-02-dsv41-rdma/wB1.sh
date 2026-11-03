#!/bin/bash
# final head 631310ae5: same-window ABAB, RDMA off / on / off / on (explicit), lcat everywhere, pfplain (1k and 4k prefill) in the first pair
cd /opt/code/intarsiaml/.worktrees/dsv41-rdma
export SPARK_DIR='~/intarsia-dsv41-rdma'
OUT=$PWD/.scratch/wB1; mkdir -p $OUT
T="INTARSIA_DSV41_SPEC_TRACE=1"
AB="INTARSIA_DSV41_RDMA=0 $T|INTARSIA_DSV41_RDMA=1 $T|INTARSIA_DSV41_RDMA=0 $T|INTARSIA_DSV41_RDMA=1 $T" STEPS_AB0="lcat pfplain" STEPS_AB1="lcat pfplain" OUT=$OUT EARLYOOM_OK="3,2 2,1" STEPS="lcat" CAT_TOKENS=256 MAX_MIN=26 scripts/dsv41-tp2/run-full.sh
