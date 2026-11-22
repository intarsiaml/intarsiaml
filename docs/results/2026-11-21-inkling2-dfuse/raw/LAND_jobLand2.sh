#!/bin/sh
# Land check, plan show only: rebuild the Metal binary (job #11488's docs_env_vars test rebuilt target/release/intarsia without
# the metal feature, so its plan show errored), then plan show Qwen3-8B / Llama against the base binary.
S=/opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch
T=/opt/code/intarsiaml/.worktrees/ink-dfuse-land
L=$S/LAND
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
cd $T || exit 1
[ "$(git rev-parse --short=9 HEAD)" = 49138c643 ] || { echo "ABORT: wrong commit"; exit 1; }
cargo build --release --bin intarsia --features metal > $L/build2.log 2>&1 || { echo "BUILD FAILED"; tail -20 $L/build2.log; exit 1; }
for m in Qwen3-8B Llama-3.1-8B-Instruct; do
  ./target/release/intarsia plan show --model ~/models/$m --backend metal > $L/plan_${m}_land.out 2> $L/plan_${m}_land.err
  if cmp -s $L/plan_${m}_base.out $L/plan_${m}_land.out && cmp -s $L/plan_${m}_base.err $L/plan_${m}_land.err; then v=byte-identical; else v=DIFFERS; fi
  echo "PLAN $m, base f55bfe8b7 vs 49138c643: $v"
done | tee $L/plan.txt
