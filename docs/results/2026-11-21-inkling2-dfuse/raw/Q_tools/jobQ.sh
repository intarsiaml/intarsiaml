#!/bin/sh
# Q (--quiet gpu,cpu; item 2, coordinator go-ahead ~09:20): decode attention-core probe, measurement only, snapshot
# .scratch/snap10 = 6368ad219 + the untracked probe example ink_attn_dump.rs (nothing shipped).
#   1. build the binary and ink_attn_dump; the probe metallibs with the backend's flags; the Swift harness;
#   2. ink_attn_dump: one backend visit per (kind, ctx) for the harness's bitwise calibration;
#   3. attnprobe: cal (bitwise vs the backend), warm16 (L's bench setting, us/visit), token (40 SWA + 8 full per token,
#      cold K / V, ms/token) for ship, the no-math kernels and the variants;
#   4. real step: box label, then the split-keys sweep 64 / 32 / 128 at short and 5k (information only: other bits).
# Pre-registered rule: docs/results/2026-11-21-inkling2-dfuse.md "Item 2: pre-registered rule".
# The body runs in its own process group; TERM kills the tree.
W=/opt/code/intarsiaml/.worktrees/ink-dfuse; S=$W/.scratch
WANT=6368ad219
L=$S/Q; mkdir -p $L
set -m
body() {
  cd $S/snap10 || exit 1
  [ "$(git rev-parse --short=9 HEAD)" = $WANT ] || { echo "ABORT: snap10 at $(git rev-parse --short=9 HEAD), not $WANT"; exit 1; }
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "ABORT: snap10 has uncommitted changes"; exit 1; }
  cmp -s crates/intarsia-backend-metal/examples/ink_attn_dump.rs $S/attn/ink_attn_dump.rs || cp $S/attn/ink_attn_dump.rs crates/intarsia-backend-metal/examples/
  [ -d target ] || cp -c -R /opt/code/intarsiaml/.worktrees/ink-metal/target target
  git ls-files -z crates kernels | xargs -0 touch
  touch crates/intarsia-backend-metal/examples/ink_attn_dump.rs
  date > $L/times.txt
  CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
  B=./target/release/intarsia
  P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
  echo "STEP build"
  cargo build --release --bin intarsia --features metal > $L/build.log 2>&1 || { echo "BUILD FAILED"; tail -30 $L/build.log; exit 1; }
  cargo build --release -p intarsia-backend-metal --example ink_attn_dump > $L/build_ex.log 2>&1 || { echo "EXAMPLE BUILD FAILED"; tail -30 $L/build_ex.log; exit 1; }
  echo "build $(date)" >> $L/times.txt
  echo "STEP probe libs + harness"
  python3 -I $S/attn/attnvar_gen.py $S/snap10/kernels/metal $L/src || exit 1
  sh $S/attn/build_libs.sh $L/src $L/libs > $L/libs.log 2>&1 || { echo "LIBS FAILED"; tail -30 $L/libs.log; exit 1; }
  tail -1 $L/libs.log
  xcrun swiftc -O -o $L/attnprobe $S/attn/attnprobe.swift > $L/swiftc.log 2>&1 || { echo "SWIFTC FAILED"; tail -30 $L/swiftc.log; exit 1; }
  echo "libs $(date)" >> $L/times.txt
  echo "STEP dump"
  ./target/release/examples/ink_attn_dump $L/dump > $L/dump.log 2>&1 || { echo "DUMP FAILED"; tail -20 $L/dump.log; exit 1; }
  echo "STEP probe"
  $S/tmo.sh 900 $L/attnprobe $L/libs $L/dump > $L/probe.txt 2> $L/probe.err; echo "EXIT probe $?"
  grep -E "^(cal|CAL|floor|bits)" $L/probe.txt
  echo "probe $(date)" >> $L/times.txt
  step_rep() {  # $1 tag, $2 split keys, $3 prompt kind (short / 5k)
    if [ $3 = 5k ]; then PR="$(cat $S/prompt4k.txt)"; else PR="$P"; fi
    env INTARSIA_INK_SPLIT_KEYS=$2 INTARSIA_METAL_PROFILE=cb:$L/cb_$1.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$PR" --max-tokens 257 > $L/cb_$1.out 2> $L/cb_$1.err
    python3 $S/cbstat.py $L/cb_$1.jsonl fb > $L/cbstat_$1.txt 2>&1
    echo "$1 sk $2 $3 $(grep -E 'span' $L/cbstat_$1.txt | tr -s ' ' | tr '\n' ' ') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cb_$1.err) md5 $(md5 -q $L/cb_$1.out)" | tee -a $L/speed.txt
  }
  echo "STEP real step: box label, split-keys sweep (information only)"
  step_rep box0 64 short; step_rep box1 64 short
  for k in short 5k; do
    i=0
    for sk in 64 32 128 128 32 64; do i=$((i + 1)); step_rep ${k}_${i}_sk$sk $sk $k; done
  done
  echo "STEP box label (end)"; step_rep box2 64 short
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
