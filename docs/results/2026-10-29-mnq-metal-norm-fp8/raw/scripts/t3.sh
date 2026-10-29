#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/mnq
B=$(ls -t .lane/target/release/deps/norm_fp8-* | grep -v '\.d$' | head -1)
for i in 1 2; do
 for v in 0 1 2; do echo "== round $i INTARSIA_METAL_NORM_FP8=$v"; INTARSIA_METAL_NORM_FP8=$v $B --ignored --nocapture --test-threads=1 timing_sequences timing_gemv_only 2>&1 | grep TIMING; done
done
