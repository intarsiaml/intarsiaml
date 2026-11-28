#!/bin/bash
# endor --load cpu, lane ink-metal-solo: CPU tests of the load change, then the Metal CLI; own target dir (never shared), sources touched.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-metal-solo
cd $W
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
find crates -name lib.rs -o -name main.rs | xargs touch
echo "[build] tests $(date +%T) git $(git rev-parse --short HEAD)"
CARGO_TARGET_DIR=$W/target cargo test --release -p intarsia-models --test inkling2_dflash2 --test recipe_ink_sampled_dflash --test inkling2_dspark --test inkling2_dspark_sampled 2>&1 | grep -E '^test |test result|error|warning: unused|panicked' | tail -40
echo "[build] tests rc ${PIPESTATUS[0]} $(date +%T)"
CARGO_TARGET_DIR=$W/target cargo test --release -p intarsia-core --lib recipe 2>&1 | grep -E 'test result|error|panicked' | tail -5
echo "[build] core recipe tests rc ${PIPESTATUS[0]} $(date +%T)"
echo "[build] metal cli $(date +%T)"
CARGO_TARGET_DIR=$W/target cargo build --release -p intarsia-cli --features metal 2>&1 | tail -3
cp $W/target/release/intarsia $W/.scratch/bin/intarsia-solo && git rev-parse --short HEAD > $W/.scratch/bin/intarsia-solo.commit && md5 $W/.scratch/bin/intarsia-solo && ls -l $W/.scratch/bin/intarsia-solo
echo "[build] done $(date +%T)"
