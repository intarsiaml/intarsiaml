#!/bin/bash
export FLEETQ_JOB=7899 FLEETQ_OWNER=claude-hrr
cd /opt/code/intarsiaml/.worktrees/hrr
for st in "A lmmlu" "B lmmlu" "A lgsm" "B lgsm" "A mmmlu" "B mmmlu" "A mgsm" "B mgsm"; do .scratch/qa/stage.sh $st; done
echo CHAIN DONE >> .scratch/qa/chain.log
