#!/bin/zsh
# Metal prefill P1, correctness burst. Run inside a fleetq slot (FLEETQ_JOB set by the caller), after
#   cp target/release/intarsia $S/intarsia-new ; cp $S/pre-target/release/intarsia $S/intarsia-pre
# pre = spike/metal 495a447 (built from `git archive` into $S/pre-src), new = branch metal-p1-gemm.
# Step 1 aborts the whole burst if the engine does not select the NAX kernels in prefill.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
W=/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7
R=$W/docs/results/raw/2026-10-14-metal-prefill-p1/full
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
PC=$W/docs/results/raw/2026-10-14-metal-phase-c
NEW=$S/intarsia-new
PRE=$S/intarsia-pre
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/select $R/decode $R/identity $R/chunk $R/accuracy

echo "== 1. kernel selection in the engine"
$PRE kernels list --backend metal > $R/select/pre-kernels-list.txt 2>&1
$NEW kernels list --backend metal > $R/select/new-kernels-list.txt 2>&1
INTARSIA_METAL_NO_METAL4=1 $NEW kernels list --backend metal > $R/select/new-nometal4-kernels-list.txt 2>&1
echo "pre lists nax kernels: $(grep -c nax $R/select/pre-kernels-list.txt) (want 0); new: $(grep -c nax $R/select/new-kernels-list.txt) (want 2); new INTARSIA_METAL_NO_METAL4=1: $(grep -c nax $R/select/new-nometal4-kernels-list.txt) (want 0)"
for B in pre new; do
  BIN=$NEW; [ $B = pre ] && BIN=$PRE
  INTARSIA_METAL_PROFILE=kernel:$R/select/$B-1k.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 \
    --temperature 0 --max-tokens 1 --prompt "$(cat $PC/prompt-1k.txt)" > $R/select/$B-1k.out 2> $R/select/$B-1k.err
  python3 -I -c "
import json, collections, sys
c = collections.Counter(json.loads(l)['op'] for l in open('$R/select/$B-1k.jsonl'))
print('$B prefill 1k kernels:', dict(c))
" | tee $R/select/$B-1k-kernels.txt
done
grep -q "matmul_bf16_nax" $R/select/new-1k-kernels.txt && grep -q "matmul_pair_bf16_nax" $R/select/new-1k-kernels.txt || { echo "ABORT: NAX kernels not selected in prefill"; exit 1; }

echo "== 2. decode dispatches per step (cb trace, 129-token decode, 364 expected on both)"
for P in short 4k; do
  for B in pre new; do
    BIN=$NEW; [ $B = pre ] && BIN=$PRE
    F=$R/decode/$B-$P
    INTARSIA_METAL_PROFILE=cb:$F.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 \
      --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $PB/prompt-$P.txt)" > $F.out 2> $F.err
    echo "$B $P: $(python3 -I $PB/b2b/period.py $F.jsonl 364 128 2>&1 | tail -1) $(grep -o '([0-9.]* tok/s)' $F.err | tail -1)"
  done
done

echo "== 3. greedy ids, pre vs new"
for N in 48 256; do
  for P in short 4k; do
    for B in pre new; do
      BIN=$NEW; [ $B = pre ] && BIN=$PRE
      $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids \
        --prompt "$(cat $PB/prompt-$P.txt)" > $R/identity/$B-$P-$N.out 2> $R/identity/$B-$P-$N.err
    done
    python3 -I $R/first_divergence.py $R/identity/pre-$P-$N.err $R/identity/new-$P-$N.err "$P $N tokens" | tee -a $R/identity/summary.txt
  done
done

echo "== 4. chunk invariance (new binary): greedy ids of the 4k prompt for several --prefill-chunk"
# 4k prompt = 4062 tokens: chunk tails 94 (128), 222 (256), 478 (512), 990 (1024) are >= 16 rows (NAX path);
# chunk 4050 leaves a 12-row final chunk, which runs on the tiled matmul_bf16 (NAX needs M >= 16)
for B in new pre; do
  BIN=$NEW; [ $B = pre ] && BIN=$PRE
  for C in 128 256 512 1024 4050; do
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids \
      --prefill-chunk $C --prompt "$(cat $PB/prompt-4k.txt)" > $R/chunk/$B-4k-c$C.out 2> $R/chunk/$B-4k-c$C.err
  done
  for C in 128 512 1024 4050; do
    python3 -I $R/first_divergence.py $R/chunk/$B-4k-c256.err $R/chunk/$B-4k-c$C.err "$B 4k: chunk 256 vs $C" | tee -a $R/chunk/summary.txt
  done
done
for C in 128 256 512; do
  for B in new pre; do
    BIN=$NEW; [ $B = pre ] && BIN=$PRE
    $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --mode prefill --max-seq 4096 --prefill-chunk $C \
      > $R/chunk/acc-$B-short-c$C.out 2> $R/chunk/acc-$B-short-c$C.err; echo "accuracy prefill $B chunk $C exit $?"
  done
done

echo "== 5. accuracy against the fp32 goldens, gate class bf16 (new; pre for side by side)"
for B in new pre; do
  BIN=$NEW; [ $B = pre ] && BIN=$PRE
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/accuracy/$B-short.out 2> $R/accuracy/$B-short.err; echo "accuracy $B short exit $?"
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $R/accuracy/$B-long.out 2> $R/accuracy/$B-long.err; echo "accuracy $B long exit $?"
done
echo done
