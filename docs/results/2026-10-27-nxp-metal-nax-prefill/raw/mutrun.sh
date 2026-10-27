#!/bin/bash
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; D=/opt/code/intarsiaml/.worktrees/nxp-mut/.nxp-out/mut; O=/opt/code/intarsiaml/.worktrees/nxp/.nxp-out/mutout; mkdir -p $O
for m in m1 m2 m3 m4; do $D/intarsia-$m kernels test --backend metal --model $M --op matmul_fp8 > $O/$m.out 2>&1; echo "$m exit $? $(grep -E 'matmul_fp8_nax' $O/$m.out | head -1 | tr -s ' ')"; done
