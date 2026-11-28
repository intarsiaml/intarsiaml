#!/bin/bash
# endor --load cpu, lane ink-dflash-cut: the Metal CLI with the opt-in DFlash2 confidence trace (INTARSIA_DFLASH_CONF_TRACE=1); own target dir.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-dflash-cut
cd $W
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
echo "[build] metal cli $(date +%T) git $(git rev-parse --short HEAD) dirty $(git status --porcelain crates | wc -l | tr -d ' ')"
CARGO_TARGET_DIR=$W/target cargo build --release -p intarsia-cli --features metal 2>&1 | grep -vE '^ +Compiling' | tail -3
cp $W/target/release/intarsia $W/.scratch/bin/intarsia-ct && md5 $W/.scratch/bin/intarsia-ct
echo "[build] done $(date +%T)"
