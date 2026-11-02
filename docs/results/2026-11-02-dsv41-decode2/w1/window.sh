#!/bin/bash
# window 1 of lane decode2: the 35B tp=2 serve check and the wo_a fork microbench with vLLM still up, then the DSv4.1 A/B of the chooseK table
cd /opt/code/intarsiaml/.worktrees/dsv41-decode2
export SPARK_DIR='~/intarsia-dsv41-decode2'
OUT=$PWD/.scratch/w1; mkdir -p $OUT
BIN_NAME=intarsia.main OUT=$OUT/tp35 scripts/dsv41-tp2/tp35-check.sh 2>&1 | tee $OUT/tp35.log
echo "== bench_wo_a_fork $(date +%T)" | tee $OUT/bench.log
ssh -n spark2 "export FLEETQ_JOB=$FLEETQ_JOB; INTARSIA_REQUIRE_GPU=1 \$HOME/intarsia-dsv41-decode2/bin/exl3_test --ignored --nocapture bench_wo_a_fork 2>&1 | tail -12" 2>&1 | tee -a $OUT/bench.log
OLD="INTARSIA_DSV41_SPEC_TRACE=1"
NEW="INTARSIA_DSV41_SPEC_TRACE=1 INTARSIA_DSV41_SPEC_ROUND_MS=30.0,36.7,41.7,47.0,52.0 INTARSIA_DSV41_SPEC_DRAFT_MS=6.0"
OUT=$OUT/dsv41 EARLYOOM_OK="3,2 2,1" STEPS=lcat CAT_TOKENS=256 MAX_MIN=17 AB="$OLD|$NEW|$OLD" scripts/dsv41-tp2/run-full.sh
