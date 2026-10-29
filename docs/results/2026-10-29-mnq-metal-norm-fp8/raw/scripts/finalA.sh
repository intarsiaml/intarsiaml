#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/mnq
mkdir -p .lane/final
.lane/gate1.sh > .lane/final/g1.log 2>&1
.lane/gate2.sh > .lane/final/g2.log 2>&1
.lane/pair.sh > .lane/final/pair.log 2>&1
echo FINALA-DONE
