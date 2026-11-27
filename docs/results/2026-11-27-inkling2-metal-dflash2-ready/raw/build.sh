#!/bin/bash
# endor --load cpu: Metal CLI of this worktree, own target dir (never shared); sources touched so cargo cannot reuse a stale crate.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-metal-dflash2-ready
cd $W
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
find crates -name '*.rs' -newer Cargo.toml -maxdepth 0 >/dev/null 2>&1
find crates -name lib.rs -o -name main.rs | xargs touch
echo "[build] cli $(date +%T) git $(git rev-parse --short HEAD)"
CARGO_TARGET_DIR=$W/target cargo build --release -p intarsia-cli --features metal 2>&1 | tail -3
cp $W/target/release/intarsia $W/.scratch/bin/intarsia-br1 && md5 $W/.scratch/bin/intarsia-br1 && ls -l $W/.scratch/bin/intarsia-br1
echo "[build] done $(date +%T)"
