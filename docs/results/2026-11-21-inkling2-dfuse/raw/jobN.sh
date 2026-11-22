#!/bin/sh
# N (--quiet gpu,cpu; coordinator brief ~07:10, item 1, measure only): why the NVFP4 experts reach 438 GB/s at 1 row.
# Snapshot .scratch/snap8 = 914738f23 (code = ink/metal d14028792 merged: the decode path is job L's, only batch.rs moved).
#   0. build; 1. box check (job L's window, 13.3 ms +- 5 %); 2. the streaming floor of one visit (.scratch/moebw.swift:
#   gate+up / down / both regions, sweep of TG count, threads, loads in flight; two dependent dispatches vs one);
#   3. the shipped kernels: gate+up / down split (INKMOE_PROBE) at 1 and 4 rows, fused-down threads 64 / 128 / 256;
#   4. 1-row variants of the shipped kernels (.scratch/moevar_gen.py + moevar.swift: lanes per row, chunks ahead,
#   simdgroups, fused-down threads; timing + max |diff| vs shipped); 5. box re-check. The body runs in its own process group; TERM kills the tree.
W=/opt/code/intarsiaml/.worktrees/ink-dfuse; S=$W/.scratch
WANT=914738f23
L=$S/N; mkdir -p $L
set -m
body() {
  cd $S/snap8 || exit 1
  [ "$(git rev-parse --short=9 HEAD)" = $WANT ] || { echo "ABORT: snap8 at $(git rev-parse --short=9 HEAD), not $WANT"; exit 1; }
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "ABORT: snap8 has uncommitted changes"; exit 1; }
  [ -d target ] || cp -c -R $W/target target
  git ls-files -z crates kernels | xargs -0 touch
  date > $L/times.txt
  CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
  B=./target/release/intarsia; X=./target/release/examples
  P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
  echo "STEP build"
  cargo build --release --bin intarsia --features metal > $L/build.log 2>&1 || { echo "BUILD FAILED"; tail -30 $L/build.log; exit 1; }
  cargo build --release -p intarsia-backend-metal --example inkmoe_bench > $L/build_ex.log 2>&1 || { echo "EXAMPLE BUILD FAILED"; tail -30 $L/build_ex.log; exit 1; }
  swiftc -O $S/moebw.swift -o $L/moebw > $L/build_swift.log 2>&1 || { echo "SWIFT BUILD FAILED"; cat $L/build_swift.log; exit 1; }
  swiftc -O $S/moevar.swift -o $L/moevar >> $L/build_swift.log 2>&1 || { echo "SWIFT BUILD FAILED (moevar)"; cat $L/build_swift.log; exit 1; }
  mkdir -p $L/var && python3 -I $S/moevar_gen.py kernels/metal $L/var > $L/var/list.txt || { echo "VARIANT GEN FAILED"; exit 1; }
  echo "build $(date)" >> $L/times.txt
  step_rep() {
    INTARSIA_METAL_PROFILE=cb:$L/cbs_$1.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 257 > $L/cbs_$1.out 2> $L/cbs_$1.err
    python3 $S/cbstat.py $L/cbs_$1.jsonl fb > $L/cbstat_$1.txt 2>&1
    echo "step $1 $(grep -E 'span|dispatches/step' $L/cbstat_$1.txt | tr -s ' ' | tr '\n' ' ') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cbs_$1.err) md5 $(md5 -q $L/cbs_$1.out)" | tee -a $L/step.txt
  }
  echo "STEP box check"
  for r in 0 1 2 3; do step_rep $r; done
  MED=$(grep -E "^step [123] " $L/step.txt | grep -oE "span ms/step median [0-9.]+" | awk '{print $4}' | sort -n | sed -n 2p)
  echo "BOX: median span of reps 1-3 = $MED ms (window 12.64-13.97)"
  if ! awk -v m="$MED" 'BEGIN { exit !(m >= 12.635 && m <= 13.965) }'; then echo "BOX SLOW: $MED ms is outside 13.3 +- 5 %; stopping"; exit 2; fi
  echo "box $(date)" >> $L/times.txt
  echo "STEP streaming floor"
  $L/moebw 2>&1 | tee $L/moebw.txt
  echo "floor $(date)" >> $L/times.txt
  echo "STEP shipped kernels"
  echo "== split, rows 1 and 4 (INKMOE_PROBE)"; INKMOE_PROBE=1 $X/inkmoe_bench nv 1 4 2>&1 | tee $L/moe_split.txt
  for n in 64 256; do echo "== fused down threads $n, 1 row"; INKMOE_DNF_THREADS=$n INKMOE_PROBE=1 $X/inkmoe_bench nv 1 2>&1 | tee $L/moe_dnf$n.txt; done
  echo "bench $(date)" >> $L/times.txt
  echo "STEP kernel variants (1 row, measurement only)"
  $L/moevar $L/var ship gu_nb2 gu_lg16 gu_lg16_nb2 gu_sg8 gu_lg4 dn_nb2 dn_ld8 dn_ld8_nb2 dn_ld16 dn_thr256 dn_ld8_thr256 ship 2>&1 | tee $L/moevar.txt
  echo "variants $(date)" >> $L/times.txt
  echo "STEP box re-check"; step_rep 4
  echo "done $(date)" >> $L/times.txt
  cat $L/times.txt
}
body &
BP=$!
cut() {
  echo "CUT ($1): killing process group $BP"
  kill -TERM -- -$BP 2>/dev/null; sleep 5; kill -KILL -- -$BP 2>/dev/null; sleep 1
  left=$(pgrep -g $BP 2>/dev/null | tr '\n' ' '); [ -n "$left" ] && echo "LEFTOVER pids in group $BP: $left" || echo "no descendants left"
  exit 143
}
trap 'cut TERM' TERM; trap 'cut INT' INT; trap 'cut HUP' HUP
wait $BP; rc=$?
echo "body exit $rc"; exit $rc
