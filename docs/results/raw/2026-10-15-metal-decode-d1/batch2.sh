#!/bin/zsh
# usage: batch2.sh <new-intarsia-bin> <glu-test-bin> <tag>
# One fleetq slot (run it with `fleetq run`): kernel-mode op traces of the base and the new binary, the fusion unit tests,
# and a 48-token greedy identity check of the new binary against the base on both prompts.
NEW=$1; TEST=$2; TAG=$3
D=/opt/code/intarsiaml/.claude/worktrees/agent-a492fe7bd7af6756c/docs/results/raw/2026-10-15-metal-decode-d1
B=/Users/josef/.local/state/claude-watch/d1bin
M=/Users/josef/models/Qwen3-8B
mkdir -p $D/trace $D/tests $D/identity
[ -f $D/trace/base-kernel.jsonl ] || {
  INTARSIA_METAL_PROFILE=kernel:$D/trace/base-kernel.jsonl $B/intarsia-base-766c723 generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $D/prompt-short.txt)" > $D/trace/base-kernel.out 2> $D/trace/base-kernel.err
  echo "base trace exit $?"
}
INTARSIA_METAL_PROFILE=kernel:$D/trace/$TAG-kernel.jsonl $NEW generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $D/prompt-short.txt)" > $D/trace/$TAG-kernel.out 2> $D/trace/$TAG-kernel.err
echo "$TAG trace exit $?"
$TEST --test-threads=1 > $D/tests/glu-$TAG.log 2>&1
echo "glu tests exit $?"
for P in short 4k; do
  [ -f $D/identity/base-$P-48.err ] || $B/intarsia-base-766c723 generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids --prompt "$(cat $D/prompt-$P.txt)" > $D/identity/base-$P-48.out 2> $D/identity/base-$P-48.err
  $NEW generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids --prompt "$(cat $D/prompt-$P.txt)" > $D/identity/$TAG-$P-48.out 2> $D/identity/$TAG-$P-48.err
  cmp -s $D/identity/base-$P-48.out $D/identity/$TAG-$P-48.out && echo "identity $P 48: stdout identical" || echo "identity $P 48: stdout DIFFERS"
done
