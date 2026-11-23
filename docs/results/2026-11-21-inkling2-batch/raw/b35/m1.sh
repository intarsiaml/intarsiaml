#!/bin/bash
# brief 35 job M1 (quiet, measure only): solo MoE experts on main $1. Build (base worktree), solo kernel trace of the real
# model (64 tokens, short prompt), moevar variants + diags (ALU-only, residency), inkmoe_bench split, moebw floor.
set -m
trap 'kill 0' TERM
O=/opt/code/intarsiaml/.worktrees/ink-moe-35/.scratch; W=/opt/code/intarsiaml/.worktrees/ink-batch; R=$W/.scratch/base; T=$W/.scratch/base-target
L=$O/m1; mkdir -p $L
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
git -C $R checkout -q --detach "$1" || exit 1
echo "tree $(git -C $R rev-parse --short HEAD); modified files: $(git -C $R status --short --untracked-files=no | wc -l | tr -d ' ')"
cd $R && git ls-files -z crates kernels recipes | xargs -0 touch
export CARGO_TARGET_DIR=$T
cargo build --release -p intarsia-cli 2>&1 | grep -E '^error|Finished'
cargo build --release -p intarsia-backend-metal --example inkmoe_bench 2>&1 | grep -E '^error|Finished'
swiftc -O $O/moevar.swift -o $L/moevar > $L/swift.log 2>&1 || { echo SWIFT FAIL; cat $L/swift.log; }
swiftc -O $O/moebw.swift -o $L/moebw >> $L/swift.log 2>&1 || echo MOEBW SWIFT FAIL
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=$T/release/intarsia
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
g() { $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens "$@"; }
echo "== warm + md5 64"; g 64 > $L/g64.out 2> $L/g64.err; echo "md5 $(md5 -q $L/g64.out) $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/g64.err)"
echo "== cb trace 64"; INTARSIA_METAL_PROFILE=cb:$L/cb.jsonl g 64 > $L/cb.out 2> $L/cb.err; grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cb.err
echo "== kernel trace 64"; INTARSIA_METAL_PROFILE=kernel:$L/k.jsonl g 64 > $L/k.out 2> $L/k.err; echo "md5 $(md5 -q $L/k.out)"
echo "== moevar"; $L/moevar $O/var ship occ alu gu_alu dn_alu xr1_dn_nb3 xr1_alu xr1_dn_nb3_alu xr1_dn_nb3_occ dn_nb3 ship 2>&1 | tee $L/moevar.txt
echo "== inkmoe_bench split"; INKMOE_PROBE=1 $T/release/examples/inkmoe_bench nv 1 16 2>&1 | tee $L/bench_nv.txt; INKMOE_PROBE=1 $T/release/examples/inkmoe_bench mx 1 2>&1 | tee $L/bench_mx.txt
echo "== moebw"; $L/moebw 2>&1 | tee $L/moebw.txt | tail -30
echo "m1 done"
