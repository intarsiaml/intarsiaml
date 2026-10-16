#!/bin/zsh
# final gates on the committed HEAD, then the chunk 256 vs 2048 cb-mode comparison. One slot.
cd /opt/code/intarsiaml-gemm-midm
D=docs/results/raw/2026-10-15-metal-gemm-midm
$D/gates.sh
echo "gates done"
$D/cb-chunk.sh 4 4062 1024
