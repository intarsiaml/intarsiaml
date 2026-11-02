#!/bin/bash
# window 4 of lane decode2: L2 prefetch against the shipped defaults, ABAB, golden on the prefetch arm
cd /opt/code/intarsiaml/.worktrees/dsv41-decode2
export SPARK_DIR='~/intarsia-dsv41-decode2'
OUT=$PWD/.scratch/w4; mkdir -p $OUT
export STEPS_AB0="golden lcat"
OUT=$OUT/dsv41 EARLYOOM_OK="3,2 2,1" STEPS=lcat CAT_TOKENS=256 MAX_MIN=20 AB="INTARSIA_DSV41_L2_PREFETCH=1|X=0|INTARSIA_DSV41_L2_PREFETCH=1|X=0" scripts/dsv41-tp2/run-full.sh
