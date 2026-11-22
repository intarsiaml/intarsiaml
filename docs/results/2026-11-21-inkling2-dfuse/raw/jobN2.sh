#!/bin/sh
# N2 (--quiet gpu,cpu; item 1, measure only): one more round of 1-row NVFP4 variants (.scratch/moevar_gen.py N2 set:
# one pair per union leader (INK_XR 1), down 2 / 3 / 4 chunks ahead, combinations), random weights for the numerics
# check, maxTotalThreadsPerThreadgroup per variant. Reuses job N's snapshot .scratch/snap8 = 914738f23 and its binary for
# the box check (no cargo build). Body in its own process group; TERM kills the tree.
W=/opt/code/intarsiaml/.worktrees/ink-dfuse; S=$W/.scratch
WANT=914738f23
L=$S/N2; mkdir -p $L
set -m
body() {
  cd $S/snap8 || exit 1
  [ "$(git rev-parse --short=9 HEAD)" = $WANT ] || { echo "ABORT: snap8 at $(git rev-parse --short=9 HEAD), not $WANT"; exit 1; }
  [ -x target/release/intarsia ] || { echo "ABORT: no binary in snap8 (job N's build)"; exit 1; }
  date > $L/times.txt
  CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
  B=./target/release/intarsia
  P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
  echo "STEP build (swift only)"
  swiftc -O $S/moevar.swift -o $L/moevar > $L/build_swift.log 2>&1 || { echo "SWIFT BUILD FAILED"; cat $L/build_swift.log; exit 1; }
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
  echo "STEP variants, pass 1"
  $L/moevar $L/var ship gu_xr1 gu_xr1_nb2 gu_xr1_lg16 dn_nb2 dn_nb3 dn_nb4 xr1_dn_nb2 xr1_dn_nb3 xr1_dn_ld8_nb2 dn_ld8 gu_lg16 gu_nb2 ship 2>&1 | tee $L/moevar1.txt
  echo "STEP variants, pass 2 (reverse order)"
  $L/moevar $L/var ship gu_nb2 gu_lg16 dn_ld8 xr1_dn_ld8_nb2 xr1_dn_nb3 xr1_dn_nb2 dn_nb4 dn_nb3 dn_nb2 gu_xr1_lg16 gu_xr1_nb2 gu_xr1 ship 2>&1 | tee $L/moevar2.txt
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
