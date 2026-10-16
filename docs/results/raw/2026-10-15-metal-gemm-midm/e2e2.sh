#!/bin/zsh
# rerun of the c2048 end-to-end after a noisy first pass: cb-mode GPU busy ABBA (4 runs per round), then localeval speed ABBA with 5 reps
cd /opt/code/intarsiaml-gemm-midm
D=docs/results/raw/2026-10-15-metal-gemm-midm
$D/cb-abba.sh 2048 3 1024 2688 3000 4062 > $D/cb-abba-c2048.txt 2>&1
echo "cb c2048 done"
$D/localeval-abba.sh localeval-c2048 2048 1024,2700,3000,4096 5
echo "localeval c2048 done"
