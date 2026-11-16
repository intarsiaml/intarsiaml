#!/bin/bash
# Lane mgemm: P1-cell replicate (qo / gu / down at M = 1024, 2048; pool 0 and 3 GiB; swizzle 0,3,4,5) + copy bandwidth,
# with the GPU utilisation sampled every 5 s. Run under `fleetq run --host endor --quiet gpu,cpu`.
W=/opt/code/intarsiaml/.worktrees/mgemm
R=$W/docs/results/2026-11-16-mgemm/raw
cd "$W" || exit 1
( while true; do echo "$(date +%T) $(ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*')"; sleep 5; done ) > "$R/repro2-ioreg.txt" 2>&1 &
SAMP=$!
"$W/target/release/examples/gemv_bw" copy > "$R/repro2-copy.txt" 2>&1
NAX_M=1024,2048 NAX_SWZ=0,3,4,5 "$W/target/release/examples/gemm_nax" bench qo gu down > "$R/repro2-pool0.txt" 2>&1
NAX_M=1024,2048 NAX_SWZ=0,3,4,5 NAX_POOL_MIB=3072 "$W/target/release/examples/gemm_nax" bench qo gu down > "$R/repro2-pool3g.txt" 2>&1
"$W/target/release/examples/gemv_bw" copy > "$R/repro2-copy-after.txt" 2>&1
kill $SAMP
