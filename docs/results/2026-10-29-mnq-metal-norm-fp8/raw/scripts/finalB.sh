#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/mnq
mkdir -p .lane/final
.lane/ab.sh single > .lane/final/ab-single.log 2>&1
.lane/ab.sh burst2 > .lane/final/ab-burst2.log 2>&1
.lane/t3.sh > .lane/final/timing.log 2>&1
echo FINALB-DONE
