#!/bin/bash
# endor --load cpu, lane ink-metal-dgemv: the gemv_g0 example with the Inkling draft cases; own target dir.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-metal-dgemv
cd $W
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
echo "[build] gemv_g0 $(date +%T) git $(git rev-parse --short HEAD) dirty $(git status --porcelain crates | wc -l | tr -d ' ')"
CARGO_TARGET_DIR=$W/target cargo build --release -p intarsia-backend-metal --example gemv_g0 2>&1 | grep -E '^error|^warning: unused|Finished' | tail -5
cp $W/target/release/examples/gemv_g0 $W/.scratch/bin/gemv_g0 && md5 $W/.scratch/bin/gemv_g0
echo "[build] done $(date +%T)"
