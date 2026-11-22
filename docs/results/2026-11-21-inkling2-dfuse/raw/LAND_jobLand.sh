#!/bin/sh
# Land check (--load cpu; coordinator ~11:15): ink/dfuse-land 49138c643 = ink/metal a63d06c97 + afaf4e8c9 (ink_ctrl_bench --probe)
# + 88d28e48a (INTARSIA_INK_DEBUG_DISPATCHES) + the results doc. Checks: build (bin + ink_ctrl_bench), docs_env_vars, the Metal
# Inkling tests (backend-metal + models on Metal), plan show Qwen3-8B (and Llama) against the base binary.
# The body runs in its own process group; TERM kills the tree.
S=/opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch
T=/opt/code/intarsiaml/.worktrees/ink-dfuse-land
WANT=49138c643
L=$S/LAND; mkdir -p $L
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
set -m
body() {
  cd $T || exit 1
  [ "$(git rev-parse --short=9 HEAD)" = $WANT ] || { echo "ABORT: land tree at $(git rev-parse --short=9 HEAD), not $WANT"; exit 1; }
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "ABORT: land tree has uncommitted changes"; exit 1; }
  [ -d target ] || cp -c -R /opt/code/intarsiaml/.worktrees/ink-metal/target target
  git ls-files -z crates kernels | xargs -0 touch
  date > $L/times.txt
  echo "STEP build"
  cargo build --release --bin intarsia --features metal > $L/build.log 2>&1 || { echo "BUILD FAILED"; tail -30 $L/build.log; exit 1; }
  cargo build --release -p intarsia-backend-metal --example ink_ctrl_bench > $L/build_ex.log 2>&1; echo "EXIT ink_ctrl_bench build $?"
  echo "build $(date)" >> $L/times.txt
  t() { tag=$1; shift; env "$@" > $L/t_$tag.log 2>&1; rc=$?
    echo "EXIT $tag $rc: $(grep -h 'test result' $L/t_$tag.log | awk '{p+=$4; f+=$6} END {print p+0 " passed, " f+0 " failed"}')"
    grep -h "panicked\|FAILED" $L/t_$tag.log | head -8; }
  {
  t docs_env_vars INTARSIA_X=0 cargo test --release -p intarsia-cli --test docs_env_vars
  t metal_ink INTARSIA_X=0 cargo test --release -p intarsia-backend-metal --test inkmoe --test inkling2_attn --test inkctrl --test inkctrl_nometal4 --test inkmix --test inkload
  t models_metal INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 --test inkling2_batch --test inkling2_mix_decode_cpu -- --test-threads 1
  } 2>&1 | tee $L/tests.txt
  echo "tests $(date)" >> $L/times.txt
  for m in Qwen3-8B Llama-3.1-8B-Instruct; do
    $S/bin/intarsia-base plan show --model ~/models/$m --backend metal > $L/plan_${m}_base.out 2> $L/plan_${m}_base.err
    ./target/release/intarsia plan show --model ~/models/$m --backend metal > $L/plan_${m}_land.out 2> $L/plan_${m}_land.err
    if cmp -s $L/plan_${m}_base.out $L/plan_${m}_land.out && cmp -s $L/plan_${m}_base.err $L/plan_${m}_land.err; then v=byte-identical; else v=DIFFERS; fi
    echo "PLAN $m, base f55bfe8b7 vs $WANT: $v"
  done | tee $L/plan.txt
  echo "done $(date)" >> $L/times.txt
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
