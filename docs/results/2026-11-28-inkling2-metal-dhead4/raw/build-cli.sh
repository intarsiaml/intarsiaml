#!/bin/bash
# endor --load cpu, lane ink-metal-dhead4: the Metal CLI on main + #560 (slot C binary); own target dir.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-metal-dhead4
cd $W
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
echo "[build] tests $(date +%T) git $(git rev-parse --short HEAD) dirty $(git status --porcelain crates recipes | wc -l | tr -d ' ')"
echo "[build] metal cli $(date +%T)"
CARGO_TARGET_DIR=$W/target cargo build --release -p intarsia-cli --features metal 2>&1 | grep -vE '^ +Compiling' | tail -3
cp $W/target/release/intarsia $W/.scratch/bin/intarsia-d4 && git rev-parse --short HEAD > $W/.scratch/bin/intarsia-d4.commit && md5 $W/.scratch/bin/intarsia-d4
echo "[build] done $(date +%T)"
