#!/bin/zsh
# final correctness checks on the committed tree: kernels test, cargo test metal, greedy identity vs the 40996bb binary,
# accuracy against the fp32 golden for both binaries. Run inside a fleetq slot.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/b2fuse
W=/opt/code/intarsiaml/.claude/worktrees/agent-a40aebdca23def5e0
R=$W/docs/results/raw/2026-10-14-metal-phase-b
O=$R/b2b
NEW=$W/target/release/intarsia
BASE=$S/intarsia-base0
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
mkdir -p $O/identity $O/checks $O/accuracy
export INTARSIA_CPU_FREE_MEM_GB=60
cd $W
cargo test -p intarsia-backend-metal --release > $O/checks/cargo-test-metal.log 2>&1; echo "cargo test metal exit $?"
$NEW kernels test --backend metal --device 0 --model $M > $O/kernels-test.out 2> $O/kernels-test.err; echo "kernels test exit $?"
for B in base new; do
  BIN=$NEW; [ $B = base ] && BIN=$BASE
  for P in short 4k; do
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids --prompt "$(cat $R/prompt-$P.txt)" > $O/identity/$B-$P.out 2> $O/identity/$B-$P.err
  done
done
for P in short 4k; do
  cmp $O/identity/base-$P.out $O/identity/new-$P.out && echo "identity $P: stdout byte-identical"
  diff <(grep '^ids:' $O/identity/base-$P.err) <(grep '^ids:' $O/identity/new-$P.err) > /dev/null && echo "identity $P: token ids identical"
done
for B in base new; do
  BIN=$NEW; [ $B = base ] && BIN=$BASE
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $O/accuracy/$B-short.out 2> $O/accuracy/$B-short.err; echo "accuracy $B short exit $?"
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $O/accuracy/$B-long.out 2> $O/accuracy/$B-long.err; echo "accuracy $B long exit $?"
done
echo done
