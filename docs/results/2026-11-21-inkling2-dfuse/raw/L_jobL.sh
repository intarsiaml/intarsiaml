#!/bin/sh
# L (--quiet gpu,cpu; coordinator brief ~06:45, measure only): a measured roofline of Metal decode at the tip, snapshot
# .scratch/snap7 = 91355d5f7 (code = afaf4e8c9 = 10cf8988d + the probe mode). Multi-dispatch timing only (no per-dispatch command
# buffers): the per-class microbenches at the real shapes, weights rotating past the system cache, plus the pure dispatch
# cost. 0. box check: the default step must be 13.3 ms +- 5 % (else BOX SLOW, stop); 1. the class benches; 2. box re-check.
W=/opt/code/intarsiaml/.worktrees/ink-dfuse; S=$W/.scratch
cd $S/snap7 || exit 1
WANT=91355d5f7
[ "$(git rev-parse --short=9 HEAD)" = $WANT ] || { echo "ABORT: snap7 at $(git rev-parse --short=9 HEAD), not $WANT"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "ABORT: snap7 has uncommitted changes"; git status --short; exit 1; }
[ -d target ] || cp -c -R $W/target target
git ls-files -z crates kernels | xargs -0 touch
L=$S/L; mkdir -p $L
date > $L/times.txt
STEP=setup; trap 'echo "CUT (SIGTERM) during step: $STEP"; cat $L/times.txt; exit 143' TERM
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=./target/release/intarsia
X=./target/release/examples
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
STEP=build
cargo build --release --bin intarsia --features metal > $L/build.log 2>&1 || { echo "BUILD FAILED"; tail -30 $L/build.log; exit 1; }
cargo build --release -p intarsia-backend-metal --example ink_p12_bench --example ink_attn_bench --example inkmoe_bench --example ink_ctrl_bench --example argmax_cost --example dispatch_cost > $L/build_ex.log 2>&1 || { echo "EXAMPLES BUILD FAILED"; tail -30 $L/build_ex.log; exit 1; }
echo "build $(date)" >> $L/times.txt
# ---- 0. box check: rep 0 warms the page cache (dropped), reps 1-3 must have a median span of 13.3 ms +- 5 %
step_rep() {  # $1 = tag
  INTARSIA_METAL_PROFILE=cb:$L/cbs_$1.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 257 > $L/cbs_$1.out 2> $L/cbs_$1.err
  python3 $S/cbstat.py $L/cbs_$1.jsonl fb > $L/cbstat_$1.txt 2>&1
  echo "step $1 $(grep -E 'span|dispatches/step' $L/cbstat_$1.txt | tr -s ' ' | tr '\n' ' ') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cbs_$1.err)" | tee -a $L/step.txt
}
STEP="box check"
for r in 0 1 2 3; do step_rep $r; done
cat $L/cbstat_2.txt
MED=$(grep -E "^step [123] " $L/step.txt | grep -oE "span ms/step median [0-9.]+" | awk '{print $4}' | sort -n | sed -n 2p)
echo "BOX: median span of reps 1-3 = $MED ms (window 12.64-13.97)"
if ! awk -v m="$MED" 'BEGIN { exit !(m >= 12.635 && m <= 13.965) }'; then echo "BOX SLOW: $MED ms is outside 13.3 +- 5 %; stopping"; cat $L/times.txt; exit 2; fi
echo "box $(date)" >> $L/times.txt
# ---- 1. the class benches (multi-dispatch command buffers, weights rotating past the system cache)
STEP="p12 bench"; echo "== ink_p12_bench (attention projections: InkInProj S 1 / 3, wo; unembed)"
$X/ink_p12_bench --rows 1 proj wo head 2>&1 | tee $L/b_p12.txt
STEP="attn bench"; echo "== ink_attn_bench (attention core)"
$X/ink_attn_bench 2>&1 | tee $L/b_attn.txt
STEP="moe bench"; echo "== inkmoe_bench (experts, 1 row)"
$X/inkmoe_bench nv mx 1 2>&1 | tee $L/b_moe.txt
STEP="ctrl bench"; echo "== ink_ctrl_bench (router, controllers; multi-dispatch mode)"
$X/ink_ctrl_bench --rows 1 2>&1 | tee $L/b_ctrl.txt
STEP="argmax"; echo "== argmax_cost"
$X/argmax_cost 2>&1 | tee $L/b_argmax.txt
STEP="dispatch cost"; echo "== dispatch_cost"
$X/dispatch_cost 2>&1 | tee $L/b_dispatch.txt
echo "benches $(date)" >> $L/times.txt
# ---- 2. box re-check (one rep)
STEP="box re-check"; step_rep 4
echo "done $(date)" >> $L/times.txt
cat $L/times.txt
