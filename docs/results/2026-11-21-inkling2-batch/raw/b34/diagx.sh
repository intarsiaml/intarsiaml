#!/bin/bash
# quiet: in_proj limiter diag. Bench InkInProj R 4/8/16 at tree $1 (base) and $2 (diag: no x unpack in ip_dot8w), A B B A.
trap 'kill 0' TERM
W=/opt/code/intarsiaml/.worktrees/ink-batch; R=$W/.scratch/base; T=$W/.scratch/base-target; O=/opt/code/intarsiaml/.worktrees/ink-batch-34/.scratch
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
export CARGO_TARGET_DIR=$T
for t in $1 $2; do git -C $R checkout -q --detach $t || exit 1; (cd $R && git ls-files -z crates kernels | xargs -0 touch && cargo build --release -p intarsia-backend-metal --example ink_mix_bench 2>&1 | grep -E '^error|Finished'); cp $T/release/examples/ink_mix_bench $O/mb_$t; done
$O/mb_$1 --reps 30 --rows 16 proj > /dev/null
for t in $1 $2 $2 $1; do echo "== tree $t"; $O/mb_$t --reps 60 --rows 4,8,16 proj | grep 'S=1'; done
echo "diagx done"
