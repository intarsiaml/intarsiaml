#!/bin/bash
# window 3 of lane decode2: 35B tp=2 serve check again (fixed script, main's binary, vLLM up), then DSv4.1 on the shipped defaults:
# ab0 shipped (golden + lcat), ab1 the defaults before this lane (env), ab2 shipped + L2 prefetch, ab3 shipped again
cd /opt/code/intarsiaml/.worktrees/dsv41-decode2
export SPARK_DIR='~/intarsia-dsv41-decode2'
OUT=$PWD/.scratch/w3; mkdir -p $OUT
BIN_NAME=intarsia.main OUT=$OUT/tp35 scripts/dsv41-tp2/tp35-check.sh 2>&1 | tee $OUT/tp35.log
OLD="INTARSIA_DSV41_LANES=0 INTARSIA_DSV41_SPEC_K=3 INTARSIA_DSV41_SPEC_ROUND_MS=31.3,38.1,42.9,47.8,52.7 INTARSIA_DSV41_SPEC_DRAFT_MS=9.0"
export STEPS_AB0="golden lcat"
OUT=$OUT/dsv41 EARLYOOM_OK="3,2 2,1" STEPS=lcat CAT_TOKENS=256 MAX_MIN=22 AB="X=0|$OLD|INTARSIA_DSV41_L2_PREFETCH=1|X=0" scripts/dsv41-tp2/run-full.sh
