#!/bin/bash
# endor --load cpu, lane ink-metal-dtopk: CPU tests (core dflash2, models inkling2_dflash2), then the Metal CLI; own target dir.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-metal-dtopk
cd $W
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
echo "[build] tests $(date +%T) git $(git rev-parse --short HEAD) dirty $(git status --porcelain crates recipes | wc -l | tr -d ' ')"
CARGO_TARGET_DIR=$W/target cargo test --release -p intarsia-core --lib dflash2 2>&1 | grep -E '^test |test result|^error|panicked' | tail -20
echo "[build] core dflash2 tests rc ${PIPESTATUS[0]} $(date +%T)"
CARGO_TARGET_DIR=$W/target cargo test --release -p intarsia-models --test inkling2_dflash2 2>&1 | grep -E 'test result|^error|panicked|FAILED' | tail -20
echo "[build] models tests rc ${PIPESTATUS[0]} $(date +%T)"
echo "[build] metal cli $(date +%T)"
CARGO_TARGET_DIR=$W/target cargo build --release -p intarsia-cli --features metal 2>&1 | grep -vE '^ +Compiling' | tail -3
cp $W/target/release/intarsia $W/.scratch/bin/intarsia-dt && git rev-parse --short HEAD > $W/.scratch/bin/intarsia-dt.commit && md5 $W/.scratch/bin/intarsia-dt
echo "[build] done $(date +%T)"
