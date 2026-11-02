#!/bin/bash
export FLEETQ_JOB=7899 FLEETQ_OWNER=claude-hrr
cd /opt/code/intarsiaml/.worktrees/hrr
for st in "A mgsm" "B mgsm"; do .scratch/qa/stage.sh $st; done
echo CHAIN2 DONE >> .scratch/qa/chain.log
