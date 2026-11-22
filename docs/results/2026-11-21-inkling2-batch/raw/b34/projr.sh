#!/bin/bash
# quiet: x-resident in_proj (tree $1): build, rustfmt, inkmix tests default and at INTARSIA_INK_PROJ_R=320/1280, then
# InkInProj us/call at 4/8/16 rows: default (gemvx) vs PROJ_R 80/160/320/640, order A B C D E E D C B A after warm-up.
trap 'kill 0' TERM
W=/opt/code/intarsiaml/.worktrees/ink-batch; R=$W/.scratch/base; T=$W/.scratch/base-target
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
git -C $R checkout -q --detach "$1" || exit 1
echo "tree $(git -C $R rev-parse --short HEAD); modified files: $(git -C $R status --short --untracked-files=no | wc -l | tr -d ' ')"
cd $R && git ls-files -z crates kernels recipes | xargs -0 touch
export CARGO_TARGET_DIR=$T
echo "rustfmt: $(rustfmt --edition 2021 --check crates/intarsia-backend-metal/src/inkmix.rs >/dev/null 2>&1 && echo clean || echo DIFF)"
cargo build --release -p intarsia-backend-metal --example ink_mix_bench 2>&1 | grep -E '^(error|warning: unused)|Finished|metal.*error'
for g in "" 320 1280; do echo "== test inkmix PROJ_R=$g"; INTARSIA_INK_PROJ_R=$g cargo test --release -p intarsia-backend-metal --test inkmix 2>&1 | grep -E 'test result|panicked|FAILED'; done
MB=$T/release/examples/ink_mix_bench
$MB --reps 30 --rows 16 proj > /dev/null
arm() { echo "== PROJ_R=$1"; INTARSIA_INK_PROJ_R=$1 $MB --reps 60 --rows 4,8,16 proj | grep -E 'S=1'; }
for a in "" 320 640 1280 1280 640 320 ""; do arm "$a"; done
echo "projr done"
