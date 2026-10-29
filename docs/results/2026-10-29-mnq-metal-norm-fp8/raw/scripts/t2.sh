#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/mnq
export CARGO_TARGET_DIR=$PWD/.lane/target
for v in 1 2; do echo "== INTARSIA_METAL_NORM_FP8=$v"; INTARSIA_METAL_NORM_FP8=$v cargo test --release -p intarsia-backend-metal --test norm_fp8 2>&1 | grep -E "^test |test result|panicked|differs"; done
