#!/bin/zsh
# one slot: pick3 A/B of the final rule, cb-mode GPU busy ABBA (3 arms), localeval speed ABBA at chunk 2048 and 256
cd /opt/code/intarsiaml-gemm-midm
D=docs/results/raw/2026-10-15-metal-gemm-midm
NAX_ROUNDS=21 NAX_M=513,600,640,700,768,800,896,900,952,1000,1024,1100,1152,1200,1280,1366,1408,1433,1536,1600,1700,1792,1850,1920,2000,2047,2048 $D/withclock.sh $D/pick3-b target/release/examples/gemm_nax pick3
echo "pick3 done"
$D/cb-abba.sh 2048 4 4062 2688 3000 3328 1024 > $D/cb-abba-c2048.txt 2>&1
echo "cb c2048 done"
$D/cb-abba.sh 256 2 4062 1024 > $D/cb-abba-c256.txt 2>&1
echo "cb c256 done"
