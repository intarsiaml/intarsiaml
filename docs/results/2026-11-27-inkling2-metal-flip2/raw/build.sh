#!/bin/bash
# endor --load cpu, lane ink-metal-flip2: CPU tests of the two flips, then the Metal CLI and the GPU prologue-split test exe; own target dir.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-metal-flip2
cd $W
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
echo "[build] tests $(date +%T) git $(git rev-parse --short HEAD) dirty $(git status --porcelain crates recipes | wc -l | tr -d ' ')"
CARGO_TARGET_DIR=$W/target cargo test --release -p intarsia-core --lib dflash2 2>&1 | grep -E '^test |test result|^error|panicked' | tail -20
echo "[build] core dflash2 tests rc ${PIPESTATUS[0]} $(date +%T)"
CARGO_TARGET_DIR=$W/target cargo test --release -p intarsia-models --test inkling2_dflash2 --test recipe_ink_sampled_dflash 2>&1 | grep -E 'test result|^error|panicked|FAILED' | tail -20
echo "[build] models tests rc ${PIPESTATUS[0]} $(date +%T)"
echo "[build] metal cli $(date +%T)"
CARGO_TARGET_DIR=$W/target cargo build --release -p intarsia-cli --features metal 2>&1 | grep -vE '^ +Compiling' | tail -3
cp $W/target/release/intarsia $W/.scratch/bin/intarsia-f2 && git rev-parse --short HEAD > $W/.scratch/bin/intarsia-f2.commit && md5 $W/.scratch/bin/intarsia-f2
echo "[build] test exe $(date +%T)"
exe=$(CARGO_TARGET_DIR=$W/target cargo test --release -p intarsia-backend-metal --test ink_prologue_split --no-run --message-format=json 2>/dev/null | python3 -c "import sys,json
for l in sys.stdin:
    try: m=json.loads(l)
    except Exception: continue
    if m.get('reason')=='compiler-artifact' and m.get('executable') and 'ink_prologue_split' in m['executable']: print(m['executable'])" | tail -1)
cp "$exe" $W/.scratch/bin/t-ps && md5 $W/.scratch/bin/t-ps
echo "[build] done $(date +%T)"
