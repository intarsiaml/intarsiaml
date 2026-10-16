#!/bin/zsh
# final end-to-end, one slot: kernel-trace A/B of the chunk-range edge (more rounds), localeval speed ABBA at chunk 2048 and 256
cd /opt/code/intarsiaml-gemm-midm
D=docs/results/raw/2026-10-15-metal-gemm-midm
$D/engine-ab5.sh 6 585 619 640
echo "engine-ab5 done"
$D/localeval-abba.sh localeval-c2048 2048 1024,2700,3000,3250,3500,4096 3
echo "localeval c2048 done"
$D/localeval-abba.sh localeval-c256 256 1024,4096 3
echo "localeval c256 done"
