#!/bin/bash
# usage: plans.sh BIN TAG
BIN=$1; TAG=$2; OUT=/opt/code/intarsiaml/.worktrees/nxp/.nxp-out/plans
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M8=$HOME/models/Qwen3-8B
$BIN plan show --model $M35 --backend metal --max-seq 20000 > $OUT/$TAG-35b-b1.txt 2>&1
$BIN plan show --model $M35 --backend metal --max-seq 20000 --max-batch 8 > $OUT/$TAG-35b-mb8.txt 2>&1
$BIN plan show --model $M8 --backend metal --max-seq 20000 > $OUT/$TAG-8b.txt 2>&1
