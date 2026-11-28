#!/bin/bash
# endor --load cpu, lane ink-metal-flip3: CPU tests (core dflash, models inkling2_dflash2), the Metal P12 test, then the Metal CLI; own target dir.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-metal-flip3
cd $W
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
echo "[build] tests $(date +%T) git $(git rev-parse --short HEAD) dirty $(git status --porcelain crates recipes | wc -l | tr -d ' ')"
CARGO_TARGET_DIR=$W/target cargo test --release -p intarsia-core --lib dflash 2>&1 | grep -E 'test result|^error|panicked|FAILED' | tail -20
echo "[build] core dflash tests rc ${PIPESTATUS[0]} $(date +%T)"
CARGO_TARGET_DIR=$W/target cargo test --release -p intarsia-models --test inkling2_dflash2 2>&1 | grep -E 'test result|^error|panicked|FAILED' | tail -20
echo "[build] models tests rc ${PIPESTATUS[0]} $(date +%T)"
echo "[build] metal cli $(date +%T)"
CARGO_TARGET_DIR=$W/target cargo build --release -p intarsia-cli --features metal 2>&1 | grep -vE '^ +Compiling' | tail -3
cp $W/target/release/intarsia $W/.scratch/bin/intarsia-f3 && git rev-parse --short HEAD > $W/.scratch/bin/intarsia-f3.commit && md5 $W/.scratch/bin/intarsia-f3
echo "[build] done $(date +%T)"
