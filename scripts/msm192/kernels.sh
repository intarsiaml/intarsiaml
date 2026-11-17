#!/bin/zsh
# Lane msm192: `intarsia kernels test --backend metal` with the knob unset and =0 (under fleetq).
set -u
W=/opt/code/intarsiaml/.worktrees/msm192; R=$W/docs/results/2026-11-16-msm192/raw
$W/target/release/intarsia kernels test --backend metal > $R/kernels-unset.out 2>&1; echo "unset exit $?"
INTARSIA_METAL_MXFP4_NAXD=0 $W/target/release/intarsia kernels test --backend metal > $R/kernels-zero.out 2>&1; echo "zero exit $?"
for f in unset zero; do echo "$f: $(tail -1 $R/kernels-$f.out) FAIL lines: $(grep -c ' FAIL' $R/kernels-$f.out)"; done
