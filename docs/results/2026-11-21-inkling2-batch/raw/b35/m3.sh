#!/bin/bash
# brief 35 job M3 (quiet): tree $1. Build + tests (inkmoe with the crossed T1 / DNU3 sweeps, rows_exact, docs_env_vars),
# inkmoe_bench nv / mx at 4 / 8 / 16 rows with pool 62 (real-text-like union) and random: DNU3 off/on A B B A, probe split;
# moevar thread-count combos on main's kernel file (var17), two passes.
set -m
trap 'kill 0' TERM
O=/opt/code/intarsiaml/.worktrees/ink-moe-35/.scratch; W=/opt/code/intarsiaml/.worktrees/ink-batch; R=$W/.scratch/base; T=$W/.scratch/base-target
L=$O/m3; mkdir -p $L
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
git -C $R checkout -q --detach "$1" || exit 1
echo "tree $(git -C $R rev-parse --short HEAD); modified files: $(git -C $R status --short --untracked-files=no | wc -l | tr -d ' ')"
cd $R && git ls-files -z crates kernels recipes | xargs -0 touch
export CARGO_TARGET_DIR=$T
echo "rustfmt: $(rustfmt --edition 2021 --check crates/intarsia-backend-metal/src/inkmoe.rs crates/intarsia-backend-metal/tests/inkmoe.rs >/dev/null 2>&1 && echo clean || echo DIFF)"
cargo build --release -p intarsia-backend-metal --example inkmoe_bench 2>&1 | grep -E '^(error|warning)|Finished'
for t in inkmoe rows_exact; do echo "== test $t"; cargo test --release -p intarsia-backend-metal --test $t -- --nocapture 2>&1 | grep -E 'test result|panicked|FAILED|INKMOE'; done
echo "== docs_env_vars"; cargo test --release -p intarsia-cli --test docs_env_vars 2>&1 | grep -E 'test result|panicked|FAILED'
MB=$T/release/examples/inkmoe_bench
$MB nv 16 > /dev/null 2>&1
arm() { echo "== DNU3=$1 pool=$2"; INTARSIA_INK_MOE_DNU3=$1 INKMOE_POOL=$2 INKMOE_PROBE=1 $MB nv 4 8 16 2>&1 | grep -E 'Auto'; }
for a in "0 62" "1 62" "1 62" "0 62" "0 " "1 " "1 " "0 "; do arm $a; done
for d in 0 1 1 0; do echo "== mx DNU3=$d pool 62"; INTARSIA_INK_MOE_DNU3=$d INKMOE_POOL=62 INKMOE_PROBE=1 $MB mx 16 2>&1 | grep Auto; done
swiftc -O $O/moevar.swift -o $L/moevar > $L/swift.log 2>&1 || { echo SWIFT FAIL; cat $L/swift.log; }
echo "== moevar pass 1"; $L/moevar $O/var17 ship dn_nb3 dn_thr256 dn_nb3_thr256 dn_nb2_thr256 xr1_dn_nb3 xr1_dn_nb3_thr256 ship 2>&1 | tee $L/moevar1.txt
echo "== moevar pass 2"; $L/moevar $O/var17 ship xr1_dn_nb3_thr256 xr1_dn_nb3 dn_nb2_thr256 dn_nb3_thr256 dn_thr256 dn_nb3 ship 2>&1 | tee $L/moevar2.txt
echo "m3 done"
