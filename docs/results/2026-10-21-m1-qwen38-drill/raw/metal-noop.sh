#!/bin/bash
# Metal no-op for PR #42 (M1, FfnBlock -> QTensor/proj() in crates/intarsia-core/src/model.rs): Qwen3-8B bf16 on endor,
# base 4be7b7d4 vs head 92794701. Runs where the endor checkout and models live; one fleetq slot:
#   fleetq run --host endor --exclusive --mem 24 --label "M1 PR #42 Metal no-op (8B base vs head)" -- \
#     bash /opt/code/intarsiaml/.claude/worktrees/agent-acebea59be77f62ac/docs/results/2026-10-21-m1-qwen38-drill/raw/metal-noop.sh
# Pass: plan, accuracy (short + long) and the 256 greedy ids identical base vs head; head kernels test 0 FAIL; head
# cargo test --workspace 0 failed. Only syntax-checked (bash -n) so far: endor was on hold when it was written. Output: $O (default /tmp/m1-metal-noop), summary at the end of stdout.
set -u
R=/opt/code/intarsiaml/.claude/worktrees/agent-acebea59be77f62ac
O=${1:-/tmp/m1-metal-noop}
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
P=$R/docs/results/raw/2026-10-14-metal-phase-b
mkdir -p $O
date +%T
for v in base:4be7b7d4 head:92794701; do
  n=${v%%:*}; rev=${v#*:}
  rm -rf $O/$n; mkdir -p $O/$n; git -C $R archive $rev | tar -x -C $O/$n
  (cd $O/$n && CARGO_TARGET_DIR=$O/target-$n cargo build --release -q -p intarsia-cli --features metal) || { echo "build $n FAILED"; exit 2; }
  echo "built $n $(date +%T)"
done
# admission first, for both trees: head's build.rs adds src/mxfp4.rs to LAUNCHER_SOURCES, which changes the launcher
# fingerprint and so invalidates cached admission results; re-admit before comparing selections
for n in base head; do
  $O/target-$n/release/intarsia kernels test --backend metal > $O/$n-kernels.out 2>&1; echo "$n kernels rc=$?"
done
for n in base head; do
  B=$O/target-$n/release/intarsia
  $B plan show --model $M --backend metal > $O/$n-plan.out 2> $O/$n-plan.err; echo "$n plan rc=$?"
  $B accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $O/$n-acc-short.out 2> $O/$n-acc-short.err; echo "$n accshort rc=$?"
  $B accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $O/$n-acc-long.out 2> $O/$n-acc-long.err; echo "$n acclong rc=$?"
  for p in short 4k; do
    $B generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 256 --ignore-eos --show-ids \
      --prompt "$(cat $P/prompt-$p.txt)" > $O/$n-gen-$p.out 2> $O/$n-gen-$p.err; echo "$n gen-$p rc=$?"
  done
done
(cd $O/head && CARGO_TARGET_DIR=$O/target-head cargo test --release --workspace > $O/head-cargo-test.log 2>&1); echo "head cargo test rc=$?"
echo "== summary"
cd $O
for f in plan.out acc-short.out acc-long.out gen-short.out gen-4k.out; do
  cmp -s base-$f head-$f && echo "$f: identical" || echo "$f: DIFFERS"
done
for p in short 4k; do
  b=$(grep '^ids:' base-gen-$p.err | md5 -q); h=$(grep '^ids:' head-gen-$p.err | md5 -q)
  [ "$b" = "$h" ] && echo "ids $p: identical (${b:0:8})" || echo "ids $p: DIFFER"
done
echo "kernels (head = base + 1 kernel / + 53 cases, matmul_mxfp4_gemv):"; tail -2 base-kernels.out; tail -2 head-kernels.out
grep "test result" head-cargo-test.log | awk '{p+=$4; f+=$6} END {print "cargo test: passed", p, "failed", f}'
date +%T
