#!/bin/zsh
# metal phase B2a GPU checks: kernels test, token identity vs the pre-change binary, per-call attention GPU-timestamp
# traces (1k / 4k / 16k context, before and after), final microbench. Run inside a fleetq slot (FLEETQ_JOB set).
# usage: run-checks.sh <worktree> <pre-binary>
WT=${1:?worktree}; PRE=${2:?pre binary}
R=$WT/docs/results/raw/2026-10-14-metal-phase-b/b2a
NEW=$WT/target/release/intarsia
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/checks $R/identity $R/trace $R/bench

$NEW kernels test --backend metal --device 0 --model $M > $R/checks/kernels-test.out 2> $R/checks/kernels-test.err; echo "kernels test exit $?"

# greedy token ids, 48 tokens: short / 1k / 4k prompts, pre vs new
for P in short 1k 4k; do
  F=$R/prompt-$P.txt; [ $P = short ] && F=$R/../prompt-short.txt; [ $P = 4k ] && F=$R/../prompt-4k.txt
  for B in pre new; do
    BIN=$NEW; [ $B = pre ] && BIN=$PRE
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --show-ids --prompt "$(cat $F)" > $R/identity/$B-$P.out 2> $R/identity/$B-$P.err
    echo "identity $B $P exit $?"
  done
done

# per-call attention GPU time: kernel mode (every launch its own command buffer), 129 forced tokens = 128 decode steps
for P in 1k 4k 16k; do
  F=$R/prompt-$P.txt; [ $P = 4k ] && F=$R/../prompt-4k.txt
  S=8192; [ $P = 16k ] && S=17000
  for B in pre new; do
    BIN=$NEW; [ $B = pre ] && BIN=$PRE
    INTARSIA_METAL_PROFILE=kernel:$R/trace/kernel-$B-$P.jsonl $BIN generate --model $M --backend metal --max-seq $S --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $F)" > $R/trace/$B-$P.out 2> $R/trace/$B-$P.err
    echo "trace $B $P exit $?"
  done
done

# microbench at the same contexts (one KV cache per layer, in-stream, GPU timestamps of the stream's command buffers)
EX=$WT/target/release/examples/attn_bw
$EX single splitkv 64 128 256 512 1024 2048 4096 8192 16384 > $R/bench/attn_bw.txt 2>&1; echo "attn_bw exit $?"
