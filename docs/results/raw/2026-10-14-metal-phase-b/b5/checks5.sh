#!/bin/zsh
# usage: checks5.sh <preBin> <newBin>
# GPU correctness checks of the B5 tree: kernels test, greedy identity vs the pre (spike/metal 62d602e) binary at 48 and 256
# tokens on both prompts, accuracy vs the fp32 goldens. Run inside a fleetq slot (FLEETQ_JOB set).
BASE=$1; NEW=$2
R=/opt/code/intarsiaml/.claude/worktrees/agent-a8c6e3392309dd202/docs/results/raw/2026-10-14-metal-phase-b
O=$R/b5
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
mkdir -p $O/identity $O/accuracy
export INTARSIA_CPU_FREE_MEM_GB=60
$NEW kernels test --backend metal --device 0 --model $M > $O/kernels-test.out 2> $O/kernels-test.err; echo "kernels test exit $?"
for N in 48 256; do
  for B in base new; do
    BIN=$NEW; [ $B = base ] && BIN=$BASE
    for P in short 4k; do
      $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids --prompt "$(cat $R/prompt-$P.txt)" > $O/identity/$B-$P-$N.out 2> $O/identity/$B-$P-$N.err
    done
  done
  for P in short 4k; do
    cmp -s $O/identity/base-$P-$N.out $O/identity/new-$P-$N.out && echo "identity $P $N tokens: stdout byte-identical" || echo "identity $P $N tokens: stdout DIFFERS"
    diff <(grep '^ids:' $O/identity/base-$P-$N.err) <(grep '^ids:' $O/identity/new-$P-$N.err) > /dev/null && echo "identity $P $N tokens: ids identical" || echo "identity $P $N tokens: ids DIFFER"
  done
done
$NEW accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $O/accuracy/new-short.out 2> $O/accuracy/new-short.err; echo "accuracy short exit $?"
$NEW accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $O/accuracy/new-long.out 2> $O/accuracy/new-long.err; echo "accuracy long exit $?"
echo done
