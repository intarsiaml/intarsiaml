#!/bin/bash
# window 5: final head 1f9b6584f on the shipped defaults, no env overrides: golden + lcat
cd /opt/code/intarsiaml/.worktrees/dsv41-decode2
export SPARK_DIR='~/intarsia-dsv41-decode2'
OUT=$PWD/.scratch/w5; mkdir -p $OUT
OUT=$OUT EARLYOOM_OK="3,2 2,1" STEPS="golden lcat" CAT_TOKENS=256 MAX_MIN=15 scripts/dsv41-tp2/run-full.sh
