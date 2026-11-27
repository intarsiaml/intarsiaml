#!/usr/bin/env bash
cd /opt/code/intarsiaml/.worktrees/ink-metal-dflash2
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
echo "[build] cli $(date +%T) head $(git rev-parse --short HEAD)"
cargo build --release -p intarsia-cli --features metal 2>&1 | tail -2
mkdir -p .scratch/bin && cp target/release/intarsia .scratch/bin/intarsia-br2 && md5 .scratch/bin/intarsia-br2
cargo test --release -p intarsia-models --test inkling2_dflash2 --test inkling2 --test inkling2_dspark --test inkling2_verify_softcap --no-run 2>&1 | grep -E "error|Finished"
cargo test --release -p intarsia-backend-metal --test dflash2 --no-run 2>&1 | grep -E "error|Finished"
echo "[build] done $(date +%T)"
