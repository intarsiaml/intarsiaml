#!/usr/bin/env bash
# build only: Metal CLI (resumable; cargo keeps its progress if the slot stops it)
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-metal-dflash2
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
echo "[build] cli $(date +%T)"
cargo build --release -p intarsia-cli --features metal 2>&1 | tail -3
mkdir -p .scratch/bin && cp target/release/intarsia .scratch/bin/intarsia-${BR:-br1} && md5 .scratch/bin/intarsia-${BR:-br1}
echo "[build] done $(date +%T)"
