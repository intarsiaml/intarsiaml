#!/bin/zsh
# After merging B6 (spike/metal 9390beb) into the branch: cargo test, kernels test, NAX selection in the 1k prefill, decode
# dispatches per step, accuracy on both goldens, greedy ids of the short prompt against the pre binary. Run in a fleetq slot.
W=/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7
R=$W/docs/results/raw/2026-10-14-metal-prefill-p1/full/post-merge
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
PC=$W/docs/results/raw/2026-10-14-metal-phase-c
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
NEW=$W/target/release/intarsia
PRE=$S/intarsia-pre
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R
cd $W
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/cargo-test-metal.txt 2>&1; echo "cargo test exit $?"
$NEW kernels test --backend metal --device 0 --model $M -v > $R/kernels-test.txt 2>&1; echo "kernels test exit $?"
INTARSIA_METAL_PROFILE=kernel:$R/select-1k.jsonl $NEW generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 \
  --max-tokens 1 --prompt "$(cat $PC/prompt-1k.txt)" > $R/select-1k.out 2> $R/select-1k.err
python3 -I -c "
import json, collections
print('new prefill 1k kernels:', dict(collections.Counter(json.loads(l)['op'] for l in open('$R/select-1k.jsonl'))))" | tee $R/select-1k-kernels.txt
INTARSIA_METAL_PROFILE=cb:$R/decode-short.jsonl $NEW generate --model $M --backend metal --max-seq 8192 --profile-every 0 \
  --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $PB/prompt-short.txt)" > $R/decode-short.out 2> $R/decode-short.err
echo "decode: $(python3 -I $PB/b2b/period.py $R/decode-short.jsonl 364 128 2>&1 | tail -1) $(grep -o '([0-9.]* tok/s)' $R/decode-short.err | tail -1)"
$NEW accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/acc-short.out 2> $R/acc-short.err; echo "accuracy short exit $?"
$NEW accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $R/acc-long.out 2> $R/acc-long.err; echo "accuracy long exit $?"
for P in short 4k; do
  for B in pre new; do
    BIN=$NEW; [ $B = pre ] && BIN=$PRE
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids \
      --prompt "$(cat $PB/prompt-$P.txt)" > $R/ids-$B-$P.out 2> $R/ids-$B-$P.err
  done
  python3 -I $W/docs/results/raw/2026-10-14-metal-prefill-p1/full/first_divergence.py $R/ids-pre-$P.err $R/ids-new-$P.err "$P 256 tokens (pre = 495a447)"
done
echo done
