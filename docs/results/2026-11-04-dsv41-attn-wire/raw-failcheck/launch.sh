#!/bin/bash
export FLEETQ_OWNER=claude-c8e51313
cd /opt/code/intarsiaml/.worktrees/dsv41-attn-wire
export SPARK_DIR=intarsia-dsv41-attnwire EARLYOOM_OK="3,2 2,1" STEPS="" MAX_MIN=120 HOOK=/opt/code/intarsiaml/.worktrees/dsv41-attn-wire/scripts/dsv41-tp2/failpaths-test.sh OUT=/opt/code/intarsiaml/.worktrees/dsv41-attn-wire/.scratch/failcheck/run1 BASE_KILLS=0
export CASES="near force ring cancel supervised golden kill kill35"
exec fleetq run --host spark1,spark2 --exclusive --mem spark1=116,spark2=116 --takes-down spark1-vllm,spark2-vllm --expect 180 --label "dsv41 attn-wire failcheck: failpaths-test.sh on merged head 18825975a (7 cases + kill35); vLLM restored after" -- /opt/code/intarsiaml/.worktrees/dsv41-attn-wire/scripts/dsv41-tp2/run-full.sh
