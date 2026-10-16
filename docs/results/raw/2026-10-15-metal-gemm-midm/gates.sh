#!/bin/zsh
# Correctness gates for the pad-aware tile pick, one fleetq slot. new = default pick, base = INTARSIA_NAX_PICK=p45 (same binary).
W=/opt/code/intarsiaml-gemm-midm
R=$W/docs/results/raw/2026-10-15-metal-gemm-midm/gates
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
PC=$W/docs/results/raw/2026-10-14-metal-phase-c
BIN=$W/target/release/intarsia
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/chunk $R/accuracy $R/identity
cd $W
cargo build --release -p intarsia-cli --features metal > $R/build.txt 2>&1
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/cargo-test-metal.txt 2>&1; echo "cargo test metal exit $?"
$BIN kernels test --backend metal --device 0 --model $M -v > $R/kernels-test.txt 2>&1; echo "kernels test exit $?"
echo "== chunk invariance: 4k prompt greedy ids, 48 tokens, chunks 128..2048, base and new"
for B in base new; do
  if [ $B = base ]; then export INTARSIA_NAX_PICK=p45; else unset INTARSIA_NAX_PICK; fi
  for C in 128 256 512 1024 2048; do
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids \
      --prefill-chunk $C --prompt "$(cat $PB/prompt-4k.txt)" > $R/chunk/$B-4k-c$C.out 2> $R/chunk/$B-4k-c$C.err
  done
  for C in 128 512 1024 2048; do
    python3 -I $R/first_divergence.py $R/chunk/$B-4k-c256.err $R/chunk/$B-4k-c$C.err "$B 4k: chunk 256 vs $C" | tee -a $R/chunk/summary.txt
  done
done
for C in 128 256 512 1024 2048; do
  python3 -I $R/first_divergence.py $R/chunk/base-4k-c$C.err $R/chunk/new-4k-c$C.err "4k chunk $C: base vs new" | tee -a $R/chunk/summary.txt
done
echo "== greedy ids 256 tokens, short + 4k, chunk 2048 (recipe default), base vs new"
for P in short 4k; do
  for B in base new; do
    if [ $B = base ]; then export INTARSIA_NAX_PICK=p45; else unset INTARSIA_NAX_PICK; fi
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids \
      --prompt "$(cat $PB/prompt-$P.txt)" > $R/identity/$B-$P-256.out 2> $R/identity/$B-$P-256.err
  done
  python3 -I $R/first_divergence.py $R/identity/base-$P-256.err $R/identity/new-$P-256.err "$P 256 tokens, base vs new" | tee -a $R/identity/summary.txt
done
echo "== accuracy (gate class bf16), new then base"
for B in new base; do
  if [ $B = base ]; then export INTARSIA_NAX_PICK=p45; else unset INTARSIA_NAX_PICK; fi
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/accuracy/$B-short.out 2> $R/accuracy/$B-short.err; echo "accuracy $B short exit $?"
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $R/accuracy/$B-long.out 2> $R/accuracy/$B-long.err; echo "accuracy $B long exit $?"
done
for C in 128 512 2048; do
  unset INTARSIA_NAX_PICK
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --mode prefill --max-seq 4096 --prefill-chunk $C \
    > $R/chunk/acc-new-short-c$C.out 2> $R/chunk/acc-new-short-c$C.err; echo "accuracy prefill new chunk $C exit $?"
done
unset INTARSIA_NAX_PICK
$BIN qualify --model $M --backend metal --bandwidth-gbs 614 --out $R/qualify > $R/qualify-console.log 2>&1; echo "qualify exit $?"
echo done
