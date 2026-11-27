#!/bin/bash
# endor --load cpu: the real-model sampled test binary of this tree (own target dir, sources touched first)
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-metal-dflash2-ready; cd $W
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
find crates -name lib.rs -o -name main.rs | xargs touch
echo "[build] test $(date +%T) git $(git rev-parse --short HEAD)"
CARGO_TARGET_DIR=$W/target cargo test --release -p intarsia-models --test inkling2_sampled_real --no-run 2>&1 | tail -4
ls -t $W/target/release/deps/inkling2_sampled_real-* | grep -v '\.d$' | head -1 | xargs md5
echo "[build] done $(date +%T)"
