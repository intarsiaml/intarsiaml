#!/bin/bash
# THE engine window (one run-full.sh call; vLLM stopped and restored once). The kernel bench showed no win, so this is the profile
# of main plus an in-situ A/B/A/B of the branch's kernels against their opt-outs:
#  0,1  main binary, chunk 512 / 2048, nsys profile (class table)   2  main, chunk 2048, nsys, INTARSIA_TP_SPIN_MS=0 (combine contention guess)
#  3-6  new binary, chunk 512: old (all opt-outs) / new (ROPE_VEC=0: its bits differ) / old / new; 7: new + fold at prefill
cd /opt/code/intarsiaml/.worktrees/dsv41-moe-other
export SPARK_DIR=~/intarsia-dsv41-attnfuse OUT=$PWD/.scratch/winA/s3-engine
D=$PWD/docs/results/2026-11-02-dsv41-prefill
OLD="INTARSIA_DSV41_HC_VEC=0 INTARSIA_DSV41_ROPE_VEC=0 INTARSIA_DSV41_ATTN_PF=0"
NEW="INTARSIA_DSV41_ROPE_VEC=0"
C5=INTARSIA_DSV41_CHUNK=512; C20=INTARSIA_DSV41_CHUNK=2048
MAIN='$HOME/intarsia-dsv41-attnfuse/bin/intarsia.main'
export CAP=33000 MAX_MIN=60 EARLYOOM_OK="3,2 2,1"
export PF_WARM=1 PF_REPS=3 LONG_PROMPTS="$D/prompt4k.txt $D/prompts/doc8k.txt"
export NSYS=0 STEPS="pfplain longp"
export BIN_AB0=$MAIN BIN_AB1=$MAIN BIN_AB2=$MAIN
export NSYS_AB0=1 NSYS_AB1=1 NSYS_AB2=1 STEPS_AB0=pfnsys STEPS_AB1=pfnsys STEPS_AB2=pfnsys
export STEPS_AB5=pfplain STEPS_AB6=pfplain
export AB="$C5|$C20|$C20 INTARSIA_TP_SPIN_MS=0|$C5 $OLD|$C5 $NEW|$C5 $OLD|$C5 $NEW|$C5 $NEW INTARSIA_DSV41_HC_FOLD_ROWS=100000"
echo "AB=$AB" > $PWD/.scratch/winA/ab-list.txt
exec scripts/dsv41-tp2/run-full.sh
