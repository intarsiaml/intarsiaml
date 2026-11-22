#!/bin/sh
# K (--quiet gpu,cpu; item 2b, coordinator brief ~09:45): what one more dependent dispatch costs in the real decode chain.
# Snapshot .scratch/snap11 = 88d28e48a (INTARSIA_INK_DEBUG_DISPATCHES, debug only) + the untracked probe example ink_attn_dump.rs.
#   1. build; bits: greedy md5 64 tokens with the env unset vs k = 200 (stop on any difference); plan show of Qwen3-8B and
#      Llama-3.1-8B against the base binary;
#   2. THE DECIDING MEASUREMENT: real decode step, k = 0 / 50 / 100 / 200 extra dispatches per step, short and 5k, after a
#      warm-up, order 0 50 100 200 200 100 50 0 per prompt; slope of span ms/step on k = cost per transition;
#   3. ride-alongs: harness chains as the engine configures them (dispcost.swift), dispatch_cost, the prologue split
#      (attnprobe token mode: ship, nopro, noclip, noconv, nobias), ink_attn_bench (L's full-512 row).
# Decision rule (fixed before timing): slope >= 2 us: reopen the dispatch target (ranked candidates to the coordinator);
# < 1.5 us: wrap up; in between: the coordinator decides. The body runs in its own process group; TERM kills the tree.
W=/opt/code/intarsiaml/.worktrees/ink-dfuse; S=$W/.scratch
WANT=88d28e48a
L=$S/K; mkdir -p $L
set -m
body() {
  cd $S/snap11 || exit 1
  [ "$(git rev-parse --short=9 HEAD)" = $WANT ] || { echo "ABORT: snap11 at $(git rev-parse --short=9 HEAD), not $WANT"; exit 1; }
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "ABORT: snap11 has uncommitted changes"; exit 1; }
  cmp -s crates/intarsia-backend-metal/examples/ink_attn_dump.rs $S/attn/ink_attn_dump.rs || cp $S/attn/ink_attn_dump.rs crates/intarsia-backend-metal/examples/
  [ -d target ] || cp -c -R $S/snap10/target target
  git ls-files -z crates kernels | xargs -0 touch
  touch crates/intarsia-backend-metal/examples/ink_attn_dump.rs
  date > $L/times.txt
  CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
  B=./target/release/intarsia
  P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
  echo "STEP build"
  cargo build --release --bin intarsia --features metal > $L/build.log 2>&1 || { echo "BUILD FAILED"; tail -30 $L/build.log; exit 1; }
  cargo build --release -p intarsia-backend-metal --example ink_attn_dump --example ink_attn_bench --example dispatch_cost > $L/build_ex.log 2>&1 || { echo "EXAMPLE BUILD FAILED"; tail -30 $L/build_ex.log; exit 1; }
  echo "build $(date)" >> $L/times.txt
  echo "STEP bits"
  $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 4096 --prompt "$P" --max-tokens 64 --temperature 0 --profile-every 0 > $L/g64_k0.out 2> $L/g64_k0.err
  env INTARSIA_INK_DEBUG_DISPATCHES=200 $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 4096 --prompt "$P" --max-tokens 64 --temperature 0 --profile-every 0 > $L/g64_k200.out 2> $L/g64_k200.err
  a=$(md5 -q $L/g64_k0.out); b=$(md5 -q $L/g64_k200.out)
  ok=$( grep -q "finish=" $L/g64_k0.err && grep -q "finish=" $L/g64_k200.err && echo ok || echo INVALID-RUN )
  echo "MD5 64 tokens ($ok): unset $a (job G: 65cde8f35a1802e3e2fadb44d94d080d), k=200 $b: $( [ $a = $b ] && echo EQUAL || echo DIFFER )" | tee $L/md5.txt
  [ $a = $b ] || { echo "STOP: the debug dispatches changed the output"; exit 3; }
  for m in Qwen3-8B Llama-3.1-8B-Instruct; do
    $S/bin/intarsia-base plan show --model ~/models/$m --backend metal > $L/plan_${m}_base.out 2> $L/plan_${m}_base.err
    $B plan show --model ~/models/$m --backend metal > $L/plan_${m}_tip.out 2> $L/plan_${m}_tip.err
    if cmp -s $L/plan_${m}_base.out $L/plan_${m}_tip.out && cmp -s $L/plan_${m}_base.err $L/plan_${m}_tip.err; then v=byte-identical; else v=DIFFERS; fi
    echo "PLAN $m, base f55bfe8b7 vs $WANT: $v"
  done | tee $L/plan.txt
  echo "bits $(date)" >> $L/times.txt
  step_rep() {  # $1 tag, $2 k, $3 prompt kind (short / 5k)
    if [ $3 = 5k ]; then PR="$(cat $S/prompt4k.txt)"; else PR="$P"; fi
    env INTARSIA_INK_DEBUG_DISPATCHES=$2 INTARSIA_METAL_PROFILE=cb:$L/cb_$1.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$PR" --max-tokens 257 > $L/cb_$1.out 2> $L/cb_$1.err
    python3 $S/cbstat.py $L/cb_$1.jsonl fb > $L/cbstat_$1.txt 2>&1
    echo "$1 k $2 $3 $(grep -E 'steps|span' $L/cbstat_$1.txt | tr -s ' ' | tr '\n' ' ') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cb_$1.err) md5 $(md5 -q $L/cb_$1.out)" | tee -a $L/speed.txt
  }
  echo "STEP real-step slope"
  step_rep box0 0 short; step_rep wu 0 short
  for kind in short 5k; do
    i=0
    for k in 0 50 100 200 200 100 50 0; do i=$((i + 1)); step_rep ${kind}_${i}_k$k $k $kind; done
  done
  python3 - $L/speed.txt <<'PY' | tee $L/slope.txt
import sys, re
rows = []
for l in open(sys.argv[1]):
    m = re.match(r"^(\S+) k (\d+) (\S+) .*dispatches/step ([0-9.]+).*span ms/step median ([0-9.]+).*\(([0-9.]+) tok/s\) md5 (\S+)", l)
    if m and not m.group(1).startswith(("box", "wu")):
        rows.append((m.group(3), int(m.group(2)), float(m.group(4)), float(m.group(5)), float(m.group(6)), m.group(7)))
for kind in ("short", "5k"):
    r = [x for x in rows if x[0] == kind]
    if len(r) < 2: continue
    ks = [x[1] for x in r]; ys = [x[3] for x in r]
    n = len(r); mk = sum(ks) / n; my = sum(ys) / n
    sxx = sum((k - mk) ** 2 for k in ks); sxy = sum((k - mk) * (y - my) for k, y in zip(ks, ys))
    b = sxy / sxx; a = my - b * mk
    res = [y - (a + b * k) for k, y in zip(ks, ys)]
    se = (sum(e * e for e in res) / (n - 2) / sxx) ** 0.5 if n > 2 else float("nan")
    by = {}
    for x in r: by.setdefault(x[1], []).append(x)
    print(f"{kind}: " + "; ".join(f"k {k}: span {', '.join('%.3f' % x[3] for x in v)} disp/step {v[0][2]:.0f} md5 {'same' if len(set(y[5] for y in v)) == 1 else 'MIXED'}" for k, v in sorted(by.items())))
    print(f"SLOPE {kind}: {b * 1000:.2f} us per dispatch (se {se * 1000:.2f}), intercept {a:.3f} ms/step; md5 across k: {'all equal' if len(set(x[5] for x in r)) == 1 else 'DIFFER'}")
PY
  echo "slope $(date)" >> $L/times.txt
  echo "STEP harness chains (engine settings) + dispatch_cost"
  xcrun swiftc -O -o $L/dispcost $S/attn/dispcost.swift > $L/swiftc_dc.log 2>&1 && $S/tmo.sh 300 $L/dispcost > $L/dispcost.txt 2>&1; echo "EXIT dispcost $?"; cat $L/dispcost.txt
  $S/tmo.sh 300 ./target/release/examples/dispatch_cost > $L/dispatch_cost.txt 2>&1; echo "EXIT dispatch_cost $?"; head -5 $L/dispatch_cost.txt
  echo "chains $(date)" >> $L/times.txt
  echo "STEP prologue split (attnprobe token mode)"
  python3 -I $S/attn/attnvar_gen.py $S/snap11/kernels/metal $L/src > /dev/null || exit 1
  sh $S/attn/build_libs.sh $L/src $L/libs > $L/libs.log 2>&1 || { echo "LIBS FAILED"; tail -20 $L/libs.log; }
  xcrun swiftc -O -o $L/attnprobe $S/attn/attnprobe.swift > $L/swiftc.log 2>&1 || { echo "SWIFTC FAILED"; tail -20 $L/swiftc.log; }
  ./target/release/examples/ink_attn_dump $L/dump > $L/dump.log 2>&1
  env PROBE_SETS=token PROBE_VARS=ship,p0,nopro,noclip,noconv,nobias $S/tmo.sh 600 $L/attnprobe $L/libs $L/dump > $L/probe.txt 2> $L/probe.err; echo "EXIT probe $?"
  grep -E "^(CAL|token|bits)" $L/probe.txt
  echo "STEP ink_attn_bench (L's bench, full-512 row)"
  $S/tmo.sh 600 ./target/release/examples/ink_attn_bench > $L/attn_bench.txt 2>&1; echo "EXIT attn_bench $?"; grep -E "floor|ctx +512 rows +1:|ctx +4096 rows +1:" $L/attn_bench.txt
  echo "STEP box label (end)"; step_rep box2 0 short
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
