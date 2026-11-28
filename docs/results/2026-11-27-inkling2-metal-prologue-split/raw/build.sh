#!/bin/bash
# endor --load cpu, lane ink-prologue2: the Metal CLI and the bitwise test exe of the prologue split (opt-in, INTARSIA_INK_PROLOGUE_SPLIT=1);
# own target dir (never shared), sources touched.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-prologue2
cd $W
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
find crates -name lib.rs -o -name main.rs | xargs touch
echo "[build] metal cli $(date +%T) git $(git rev-parse --short HEAD)"
CARGO_TARGET_DIR=$W/target cargo build --release -p intarsia-cli --features metal 2>&1 | grep -vE '^ +Compiling' | tail -5
cp $W/target/release/intarsia $W/.scratch/bin/intarsia-ps && git rev-parse --short HEAD > $W/.scratch/bin/intarsia-ps.commit && md5 $W/.scratch/bin/intarsia-ps
echo "[build] test exe $(date +%T)"
exe=$(CARGO_TARGET_DIR=$W/target cargo test --release -p intarsia-backend-metal --test ink_prologue_split --no-run --message-format=json 2>/dev/null | python3 -c "import sys,json
for l in sys.stdin:
    try: m=json.loads(l)
    except Exception: continue
    if m.get('reason')=='compiler-artifact' and m.get('executable') and 'ink_prologue_split' in m['executable']: print(m['executable'])" | tail -1)
echo "[build] test exe: $exe"
cp "$exe" $W/.scratch/bin/t-ps && md5 $W/.scratch/bin/t-ps
echo "[build] done $(date +%T)"
