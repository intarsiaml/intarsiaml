#!/bin/zsh
R=/opt/code/intarsiaml/docs/results/raw/2026-10-14-metal-phase-b
NEW=/opt/code/intarsiaml/target/release/intarsia
M=/Users/josef/models/Qwen3-8B
cd /opt/code/intarsiaml
mkdir -p $R/checks $R/engine-op-table
export INTARSIA_CPU_FREE_MEM_GB=60
cargo test -p intarsia-backend-metal --release > $R/checks/cargo-test-metal.log 2>&1; echo "cargo test exit $?"
$NEW kernels test --backend metal --device 0 --model $M > $R/checks/kernels-test.out 2> $R/checks/kernels-test.err; echo "kernels test exit $?"
for N in 1 65; do
  $NEW generate --model $M --backend metal --max-seq 8192 --profile-every 1 --op-table --temperature 0 --max-tokens $N --ignore-eos --prompt "$(cat $R/prompt-short.txt)" > $R/engine-op-table/short-$N.out 2> $R/engine-op-table/short-$N.err
done
$NEW qualify --model $M --backend metal --quick-only --out $R/qualify-quick > $R/checks/qualify-quick-console.log 2>&1; echo "qualify exit $?"
