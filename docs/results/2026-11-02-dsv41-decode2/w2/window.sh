#!/bin/bash
# window 2 of lane decode2: 35B tp=2 serve check on main's binary (vLLM up), then DSv4.1: base | lanes | lanes + SPEC_K=4 with the measured table
cd /opt/code/intarsiaml/.worktrees/dsv41-decode2
export SPARK_DIR='~/intarsia-dsv41-decode2'
OUT=$PWD/.scratch/w2; mkdir -p $OUT
BIN_NAME=intarsia.main OUT=$OUT/tp35 scripts/dsv41-tp2/tp35-check.sh 2>&1 | tee $OUT/tp35.log
T="INTARSIA_DSV41_SPEC_ROUND_MS=31.1,36.5,41.2,46.0,50.8 INTARSIA_DSV41_SPEC_DRAFT_MS=5.5"
export STEPS_AB1="golden lcat" STEPS_AB2="golden lcat"
OUT=$OUT/dsv41 EARLYOOM_OK="3,2 2,1" STEPS=lcat CAT_TOKENS=256 MAX_MIN=22 AB="INTARSIA_DSV41_SPEC_TRACE=1|INTARSIA_DSV41_SPEC_TRACE=1 INTARSIA_DSV41_LANES=1|INTARSIA_DSV41_SPEC_TRACE=1 INTARSIA_DSV41_LANES=1 INTARSIA_DSV41_SPEC_K=4 $T" scripts/dsv41-tp2/run-full.sh
