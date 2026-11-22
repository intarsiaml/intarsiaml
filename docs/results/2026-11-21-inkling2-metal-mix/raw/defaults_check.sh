#!/bin/bash
set -o pipefail
cd /opt/code/intarsiaml/.worktrees/ink-m-mix
echo "HEAD $(git rev-parse --short HEAD)"
cargo build --release -p intarsia-cli --features metal 2>&1 | grep -E "^error|Finished" || exit 1
cargo build --release -p intarsia-backend-metal --example ink_mix_bench --tests 2>&1 | grep -E "^error|Finished" || exit 1
echo "== rows-exact tests"
cargo test --release -p intarsia-backend-metal --test inkmix 2>&1 | grep -E "^test |test result|panicked"
echo "== admission"
for op in conv_mix attn_res_mix sconv_res loop_blend; do ./target/release/intarsia kernels test --backend metal --op $op 2>&1 | grep -E "ADMITTED|REJECTED|FAIL|NO CASES"; done
for name in before after; do
  B=./target/release/examples/ink_mix_bench; [ $name = before ] && B=.scratch/ink_mix_bench_before
  echo "== bench $name"
  INK_BENCH_ONLY=ConvMix $B --reps 6 --rows 1,16 mix | grep -E "plain|blend S=3"
  INK_BENCH_ONLY=AttnResMix $B --reps 6 --rows 1,16 mix | grep "N=4"
  INK_BENCH_ONLY=LoopBlend $B --reps 6 --rows 1,16 mix | grep LoopBlend
done
