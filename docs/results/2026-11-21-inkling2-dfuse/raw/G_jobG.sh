#!/bin/sh
# G (--quiet gpu,cpu; coordinator go-ahead ~07:55): the gates of INTARSIA_INK_MOE_T1 (fcd05b172, default off) on Metal,
# snapshot .scratch/snap9. One binary, the switch by env: off = the tip, on = INTARSIA_INK_MOE_T1=1.
#   1. box label (a label, not a stop: the speed rule); 2. greedy md5 64 / 256 off vs on (any difference: stop);
#   3. tests: inkmoe (rows-exact with the switch off and on, cross-kernel), G1 tiny + inkling2_batch on Metal with it on;
#   4. admission moe_experts off / on; 5. G2 (both modes) + G3 off and on, self-goldens byte-compared; 6. plan show;
#   7. speed: warm-up, then ABAB in this slot (short 4 pairs, 5k 3 pairs), per-pair span and tok/s differences.
# Decision rule (fixed before timing): per-pair median gain >= 0.3 ms per token, warm steady state.
# The body runs in its own process group; TERM kills the tree.
W=/opt/code/intarsiaml/.worktrees/ink-dfuse; S=$W/.scratch
WANT=fcd05b172
L=$S/G; mkdir -p $L
set -m
body() {
  cd $S/snap9 || exit 1
  [ "$(git rev-parse --short=9 HEAD)" = $WANT ] || { echo "ABORT: snap9 at $(git rev-parse --short=9 HEAD), not $WANT"; exit 1; }
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "ABORT: snap9 has uncommitted changes"; exit 1; }
  [ -d target ] || cp -c -R $S/snap8/target target
  git ls-files -z crates kernels | xargs -0 touch
  date > $L/times.txt
  CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
  B=./target/release/intarsia
  P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
  ON=INTARSIA_INK_MOE_T1=1; OFF=INTARSIA_INK_MOE_T1=0
  echo "STEP build"
  cargo build --release --bin intarsia --features metal > $L/build.log 2>&1 || { echo "BUILD FAILED"; tail -30 $L/build.log; exit 1; }
  echo "build $(date)" >> $L/times.txt
  step_rep() {  # $1 tag, $2 env, $3 prompt kind (short / 5k)
    if [ $3 = 5k ]; then PR="$(cat $S/prompt4k.txt)"; else PR="$P"; fi
    env $2 INTARSIA_METAL_PROFILE=cb:$L/cb_$1.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$PR" --max-tokens 257 > $L/cb_$1.out 2> $L/cb_$1.err
    python3 $S/cbstat.py $L/cb_$1.jsonl fb > $L/cbstat_$1.txt 2>&1
    echo "$1 $(grep -E 'span' $L/cbstat_$1.txt | tr -s ' ' | tr '\n' ' ') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cb_$1.err) served $(grep -c 'served by ink_moe_gu1' $L/cb_$1.err) md5 $(md5 -q $L/cb_$1.out)" | tee -a $L/speed.txt
  }
  echo "STEP box label"
  step_rep box0 $OFF short; step_rep box1 $OFF short
  MED=$(grep -E "^box1 " $L/speed.txt | grep -oE "span ms/step median [0-9.]+" | awk '{print $4}')
  awk -v m="$MED" 'BEGIN { exit !(m >= 12.635 && m <= 13.965) }' && echo "BOX LABEL: OK ($MED ms, cold-like)" || echo "BOX LABEL: SLOW ($MED ms, warm or loaded)"
  echo "STEP greedy md5 64 / 256"
  for n in 64 256; do for m in off on; do
    if [ $m = on ]; then E=$ON; else E=$OFF; fi
    env $E $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 4096 --prompt "$P" --max-tokens $n --temperature 0 --profile-every 0 > $L/g${n}_$m.out 2> $L/g${n}_$m.err
  done
    ok=$( [ -s $L/g${n}_off.out ] && [ -s $L/g${n}_on.out ] && grep -q "finish=" $L/g${n}_off.err && grep -q "finish=" $L/g${n}_on.err && echo ok || echo INVALID-RUN )
    a=$(md5 -q $L/g${n}_off.out); b=$(md5 -q $L/g${n}_on.out)
    echo "MD5 $n tokens ($ok): off $a, on $b: $( [ $a = $b ] && echo EQUAL || echo DIFFER ); served on $(grep -c 'served by ink_moe_gu1' $L/g${n}_on.err) off $(grep -c 'served by ink_moe_gu1' $L/g${n}_off.err)" | tee -a $L/md5.txt
  done
  grep -q DIFFER $L/md5.txt && { echo "STOP: greedy md5 differs (a bug in a bitwise change)"; exit 3; }
  echo "md5 $(date)" >> $L/times.txt
  echo "STEP tests"
  t() { tag=$1; shift; env "$@" > $L/t_$tag.log 2>&1; rc=$?
    echo "EXIT $tag $rc: $(grep -h 'test result' $L/t_$tag.log | awk '{p+=$4; f+=$6} END {print p+0 " passed, " f+0 " failed"}')"
    grep -h "panicked\|FAILED\|INKMOE" $L/t_$tag.log | head -8; }
  {
  t inkmoe INTARSIA_X=0 cargo test --release -p intarsia-backend-metal --test inkmoe -- --nocapture
  t models_on $ON INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 --test inkling2_batch -- --nocapture
  } 2>&1 | tee $L/tests.txt
  echo "tests $(date)" >> $L/times.txt
  echo "STEP admission"
  for m in off on; do
    if [ $m = on ]; then E=$ON; else E=$OFF; fi
    env $E $B kernels test --backend metal --op moe_experts > $L/kt_$m.log 2>&1; echo "EXIT kt_$m $?" >> $L/kt_$m.log
    echo "== admission $m"; grep -h "situ\|EXIT" $L/kt_$m.log | head -8
  done | tee $L/admission.txt
  echo "admission $(date)" >> $L/times.txt
  echo "STEP G2 / G3 off and on"
  for g in short long; do
    for m in off on; do
      if [ $m = on ]; then E=$ON; else E=$OFF; fi
      kl=$( [ $g = short ] && echo 4e-3 || echo 6e-3 )
      env $E $B accuracy --model $CK --backend metal --max-seq 4096 --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $L/g2-$g-$m-self.safetensors > $L/g2-$g-$m.log 2>&1; echo EXIT $? >> $L/g2-$g-$m.log
      kl=$( [ $g = short ] && echo 1.05e-2 || echo 1.65e-2 )
      env $E $B accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --golden /opt/code/intarsiaml/golden/inkling2-mini-vllm-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $L/g3-$g-$m-self.safetensors > $L/g3-$g-$m.log 2>&1; echo EXIT $? >> $L/g3-$g-$m.log
    done
    for k in g2 g3; do
      echo "== $k-$g on: $(grep -h '^OVERALL' $L/$k-$g-on.log | tr -s ' ' | cut -c1-90 | tr '\n' '|') $(tail -1 $L/$k-$g-on.log)"
      echo "   self-golden off vs on: $(python3 -I $S/stcmp.py $L/$k-$g-off-self.safetensors $L/$k-$g-on-self.safetensors 2>&1)"
    done | tee -a $L/gates.txt
  done
  echo "gates $(date)" >> $L/times.txt
  echo "STEP plan show"
  for m in Qwen3-8B Llama-3.1-8B-Instruct; do
    $S/bin/intarsia-base plan show --model ~/models/$m --backend metal > $L/plan_${m}_base.out 2> $L/plan_${m}_base.err
    $B plan show --model ~/models/$m --backend metal > $L/plan_${m}_tip.out 2> $L/plan_${m}_tip.err
    if cmp -s $L/plan_${m}_base.out $L/plan_${m}_tip.out && cmp -s $L/plan_${m}_base.err $L/plan_${m}_tip.err; then v=byte-identical; else v=DIFFERS; fi
    echo "PLAN $m, base f55bfe8b7 vs $WANT: $v"
  done | tee $L/plan.txt
  echo "STEP speed ABAB"
  step_rep wu_off $OFF short; step_rep wu_on $ON short
  for i in 1 2 3 4; do
    if [ $((i % 2)) = 1 ]; then step_rep s${i}_off $OFF short; step_rep s${i}_on $ON short; else step_rep s${i}_on $ON short; step_rep s${i}_off $OFF short; fi
  done
  for i in 1 2 3; do
    if [ $((i % 2)) = 1 ]; then step_rep l${i}_off $OFF 5k; step_rep l${i}_on $ON 5k; else step_rep l${i}_on $ON 5k; step_rep l${i}_off $OFF 5k; fi
  done
  python3 - $L/speed.txt <<'PY' | tee $L/speed_summary.txt
import sys, re, statistics as st
rows = {}
for l in open(sys.argv[1]):
    m = re.match(r"^(\S+) .*span ms/step median ([0-9.]+).*\(([0-9.]+) tok/s\) served (\d+)", l)
    if m: rows[m.group(1)] = (float(m.group(2)), float(m.group(3)), int(m.group(4)))
for kind, n in (("s", 4), ("l", 3)):
    d = [rows[f"{kind}{i}_off"][0] - rows[f"{kind}{i}_on"][0] for i in range(1, n + 1) if f"{kind}{i}_off" in rows and f"{kind}{i}_on" in rows]
    t = [rows[f"{kind}{i}_on"][1] - rows[f"{kind}{i}_off"][1] for i in range(1, n + 1) if f"{kind}{i}_off" in rows and f"{kind}{i}_on" in rows]
    sv = [rows[f"{kind}{i}_on"][2] for i in range(1, n + 1) if f"{kind}{i}_on" in rows], [rows[f"{kind}{i}_off"][2] for i in range(1, n + 1) if f"{kind}{i}_off" in rows]
    name = "short" if kind == "s" else "5k"
    if d:
        print(f"{name}: per-pair span gain off - on (ms/step) {['%.3f' % x for x in d]} median {st.median(d):.3f}; tok/s on - off {['%.2f' % x for x in t]} median {st.median(t):.2f}; served on {sv[0]} off {sv[1]}")
        print(f"DECISION {name}: {'GO (>= 0.3 ms)' if st.median(d) >= 0.3 else 'below 0.3 ms'}")
PY
  echo "STEP box label (end)"; step_rep box2 $OFF short
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
