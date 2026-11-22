#!/bin/sh
# P (--quiet gpu,cpu; coordinator go-ahead ~06:15, measure only): the ConvMix row-kernel probe on the tip, snapshot
# .scratch/snap6 = afaf4e8c9 (10cf8988d = ink/metal 1c1ee202c with the NE fix, + ink_ctrl_bench --probe). No shipped-kernel
# change. Order: 1. re-trace the controllers (NE 3 now) + the normal step; 2. the row kernel alone per call site (plain and
# with the pair stage, S 1 / 3), cached and after an evict stream; 3. the ceiling over all 96 calls (stop rule: < 0.3 ms).
W=/opt/code/intarsiaml/.worktrees/ink-dfuse; S=$W/.scratch
cd $S/snap6 || exit 1
WANT=afaf4e8c9
[ "$(git rev-parse --short=9 HEAD)" = $WANT ] || { echo "ABORT: snap6 at $(git rev-parse --short=9 HEAD), not $WANT"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "ABORT: snap6 has uncommitted changes"; git status --short; exit 1; }
[ -d target ] || cp -c -R $W/target target
# build hazard: a cloned target's outputs can be newer than this tree's sources; touch them all so everything rebuilds
git ls-files -z crates kernels | xargs -0 touch
L=$S/P; mkdir -p $L
date > $L/times.txt
STEP=setup; trap 'echo "CUT (SIGTERM) during step: $STEP"; cat $L/times.txt; exit 143' TERM
RAW=docs/results/2026-11-21-inkling2-dfuse/raw
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=./target/release/intarsia
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
# ---- 0. build
STEP=build
cargo build --release --bin intarsia --features metal > $L/build.log 2>&1 || { echo "BUILD FAILED"; tail -30 $L/build.log; exit 1; }
cargo build --release -p intarsia-backend-metal --example ink_ctrl_bench > $L/build_bench.log 2>&1 || { echo "BENCH BUILD FAILED"; tail -30 $L/build_bench.log; exit 1; }
echo "build $(date)" >> $L/times.txt
# ---- 1. controllers on the tip (the defaults), per dispatch; the normal step for reference
STEP="per-dispatch trace"
INTARSIA_TOKEN_FEEDBACK=0 INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$L/disp.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 33 > $L/disp.out 2> $L/disp.err
echo "== per-dispatch, tip defaults (run ok: $( [ -s $L/disp.out ] && grep -q 'finish=' $L/disp.err && echo yes || echo NO))"
python3 $S/disp.py $L/disp.jsonl | tee $L/disp_summary.txt
python3 $RAW/calltype.py $L/disp.jsonl | tee $L/calltype.txt
STEP="normal step"
for rep in 1 2 3; do
  INTARSIA_METAL_PROFILE=cb:$L/cbs_$rep.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 257 > $L/cbs_$rep.out 2> $L/cbs_$rep.err
  echo "short $rep $(python3 $S/cbstat.py $L/cbs_$rep.jsonl fb 2>&1 | grep -E 'span|dispatches/step' | tr -s ' ' | tr '\n' ' ') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cbs_$rep.err) md5 $(md5 -q $L/cbs_$rep.out)" | tee -a $L/step.txt
done
echo "trace+step $(date)" >> $L/times.txt
# ---- 2. the row kernel alone per call site, cached and after an evict stream (96 / 256 MB)
STEP="bench probe"
./target/release/examples/ink_ctrl_bench --probe --rows 1 --evict 0,96,256 > $L/probe.md 2> $L/probe.err; echo "EXIT probe $?"
cat $L/probe.md; tail -3 $L/probe.err
echo "probe $(date)" >> $L/times.txt
# ---- 3. the ceiling over all 96 calls (stop rule: under 0.3 ms per token)
python3 $RAW/ceil.py $L/disp.jsonl $L/probe.md | tee $L/ceil.txt
echo "done $(date)" >> $L/times.txt
cat $L/times.txt
