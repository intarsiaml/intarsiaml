#!/bin/zsh
# Metal P3: (re-)record the Metal accuracy baselines and noise envelopes at the new default chunk (2048), run the long-golden
# prefix-reuse gate at chunk 256 (the only chunk where the 8B long golden can run it), and a warm-vs-cold serve check on a
# 4k prompt at 2048 and 256. One fleetq slot. Binary = stamped P3 build (head=e8be186 code-tree=6c2d16b008e2 dirty=no).
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
W=/opt/code/intarsiaml/.claude/worktrees/agent-af7ec7dd89422c6a9
R=$W/docs/results/raw/2026-10-14-metal-prefill-p3/baselines
RAW=$W/docs/results/raw
P3=$S/p3/intarsia-p3
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R $W/golden/baselines
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 45 --label "metal P3: baselines, envelope, reuse gate, warm-vs-cold serve check" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
SB=$W/golden/baselines

echo "== 1. short golden: baseline, envelope (chunk=64 splits 3 of 8 prompts), self-check"
SF=(--mode both --decode-positions 16 --max-seq 4096)
$P3 accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors $SF --write-baseline $SB/qwen3-8b-fp32.metal.baseline.json > $R/short-baseline.out 2> $R/short-baseline.err; echo "short baseline exit $?"
$P3 accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors $SF --noise-ref $SB/qwen3-8b-fp32.metal.baseline.json --noise-perturb chunk=64 \
  --write-noise-envelope $SB/qwen3-8b-fp32.metal.envelope.json --allow-same-code > $R/short-envelope.out 2> $R/short-envelope.err; echo "short envelope exit $?"
$P3 accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors $SF --baseline $SB/qwen3-8b-fp32.metal.baseline.json \
  --noise-envelope $SB/qwen3-8b-fp32.metal.envelope.json --allow-same-code --pre-change e8be186 > $R/short-selfcheck.out 2> $R/short-selfcheck.err; echo "short selfcheck exit $?"
grep -E "verdict" $R/short-selfcheck.out | cut -c1-300

echo "== 2. long golden: baseline, envelope (chunk=256 splits 2 of 10 prompts), self-check"
LF=(--mode both --decode-positions 64 --max-seq 8192)
$P3 accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors $LF --write-baseline $SB/qwen3-8b-long-fp32.metal.baseline.json > $R/long-baseline.out 2> $R/long-baseline.err; echo "long baseline exit $?"
$P3 accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors $LF --noise-ref $SB/qwen3-8b-long-fp32.metal.baseline.json --noise-perturb chunk=256 \
  --write-noise-envelope $SB/qwen3-8b-long-fp32.metal.envelope.json --allow-same-code > $R/long-envelope.out 2> $R/long-envelope.err; echo "long envelope exit $?"
$P3 accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors $LF --baseline $SB/qwen3-8b-long-fp32.metal.baseline.json \
  --noise-envelope $SB/qwen3-8b-long-fp32.metal.envelope.json --allow-same-code --pre-change e8be186 > $R/long-selfcheck.out 2> $R/long-selfcheck.err; echo "long selfcheck exit $?"
grep -E "verdict" $R/long-selfcheck.out | cut -c1-300

echo "== 3. long golden --mode all (as qualify's acc-long) at chunk 256 and at the default 2048"
for C in 256 default; do
  if [ $C = default ]; then X=(); else X=(--prefill-chunk $C); fi
  $P3 accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --mode all --decode-positions 64 --max-seq 8192 $X > $R/long-all-c$C.out 2> $R/long-all-c$C.err; echo "long all chunk $C exit $?"
  grep -E "prefill chunk|reuse gate|prefix-reuse|floor verdict" $R/long-all-c$C.out $R/long-all-c$C.err | cut -c1-300
done

echo "== 4. warm-vs-cold prefix reuse through serve, 4k prompt, greedy 48 tokens"
python3 -I $W/docs/results/raw/2026-10-14-metal-prefill-p3/reuse_serve.py $P3 $M $RAW/2026-10-14-metal-phase-c/prompt-4k.txt $R/reuse-serve 2>&1 | tee $R/reuse-serve.log
echo done
