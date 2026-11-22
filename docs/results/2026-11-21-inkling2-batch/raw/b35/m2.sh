#!/bin/bash
# brief 35 job M2 (quiet): tree $1 (T1 port). Build cli (metal) + inkmoe_bench, Metal tests inkmoe / rows_exact, md5 64/256
# T1 off vs on, solo cb + kernel traces (T1 off), moevar LUT decode variants (two passes, opposite order).
set -m
trap 'kill 0' TERM
O=/opt/code/intarsiaml/.worktrees/ink-moe-35/.scratch; W=/opt/code/intarsiaml/.worktrees/ink-batch; R=$W/.scratch/base; T=$W/.scratch/base-target
L=$O/m2; mkdir -p $L
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
git -C $R checkout -q --detach "$1" || exit 1
echo "tree $(git -C $R rev-parse --short HEAD); modified files: $(git -C $R status --short --untracked-files=no | wc -l | tr -d ' ')"
cd $R && git ls-files -z crates kernels recipes | xargs -0 touch
export CARGO_TARGET_DIR=$T
cargo build --release -p intarsia-cli --features metal 2>&1 | grep -E '^error|Finished'
for t in inkmoe rows_exact; do echo "== test $t"; cargo test --release -p intarsia-backend-metal --test $t 2>&1 | grep -E 'test result|panicked|FAILED'; done
swiftc -O $O/moevar.swift -o $L/moevar > $L/swift.log 2>&1 || { echo SWIFT FAIL; cat $L/swift.log; }
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=$T/release/intarsia
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
g() { $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens "$@"; }
for n in 64 256; do for t1 in 0 1; do INTARSIA_INK_MOE_T1=$t1 g $n > $L/g${n}_t$t1.out 2> $L/g${n}_t$t1.err
  echo "md5 $n T1=$t1 $(md5 -q $L/g${n}_t$t1.out) $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/g${n}_t$t1.err)"; done; done
echo "== cb trace 64"; INTARSIA_METAL_PROFILE=cb:$L/cb.jsonl g 64 > $L/cb.out 2> $L/cb.err; grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cb.err
echo "== kernel trace 64"; INTARSIA_METAL_PROFILE=kernel:$L/k.jsonl g 64 > $L/k.out 2> $L/k.err; echo "md5 $(md5 -q $L/k.out)"
echo "== moevar pass 1"; $L/moevar $O/var ship lut xr1_dn_nb3 xr1_dn_nb3_lut xr1_alu xr1_lut_alu ship 2>&1 | tee $L/moevar1.txt
echo "== moevar pass 2"; $L/moevar $O/var ship xr1_lut_alu xr1_alu xr1_dn_nb3_lut xr1_dn_nb3 lut ship 2>&1 | tee $L/moevar2.txt
echo "m2 done"
