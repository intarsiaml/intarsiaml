#!/bin/zsh
# Correctness gates, pre (spike/metal 766c723 binary) against new (metal-p45-gemm merged tree). One fleetq slot.
#   1 kernel selection, 2 decode dispatches/step (364), 3 greedy ids 48/256 short + 4k, 4 chunk invariance (ids for chunks
#   128/256/512/1024/2048, new and pre, 4k prompt), 5 accuracy against both goldens (+ prefill at three chunks).
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p45
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
R=$W/docs/results/raw/2026-10-15-metal-prefill-p45/gates
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
PC=$W/docs/results/raw/2026-10-14-metal-phase-c
NEW=$S/intarsia-t2
PRE=$S/intarsia-pre
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/select $R/decode $R/identity $R/chunk $R/accuracy
pgrep -fl zoom.us > $R/zoom.txt 2>&1
date > $R/start.txt

echo "== 1. kernel selection"
$PRE kernels list --backend metal > $R/select/pre-kernels-list.txt 2>&1
$NEW kernels list --backend metal > $R/select/new-kernels-list.txt 2>&1
for B in pre new; do
  BIN=$NEW; [ $B = pre ] && BIN=$PRE
  INTARSIA_METAL_PROFILE=kernel:$R/select/$B-1k.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 \
    --temperature 0 --max-tokens 1 --prompt "$(cat $PC/prompt-1k.txt)" > $R/select/$B-1k.out 2> $R/select/$B-1k.err
  python3 -I -c "
import json, collections
c = collections.Counter(json.loads(l)['op'] for l in open('$R/select/$B-1k.jsonl'))
print('$B prefill 1k kernels:', dict(c))
" | tee $R/select/$B-1k-kernels.txt
done

echo "== 2. decode dispatches per step (cb trace, 129-token decode)"
for P in short 4k; do
  for B in pre new; do
    BIN=$NEW; [ $B = pre ] && BIN=$PRE
    F=$R/decode/$B-$P
    INTARSIA_METAL_PROFILE=cb:$F.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 \
      --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $PB/prompt-$P.txt)" > $F.out 2> $F.err
    echo "$B $P: step_ms busy_ms idle_ms dispatches/step: $(python3 -I $R/period.py $F.jsonl 364 128 2>&1 | tail -1) $(grep -o '([0-9.]* tok/s)' $F.err | tail -1)"
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
    python3 -I $R/first_divergence.py $R/identity/pre-$P-$N.err $R/identity/new-$P-$N.err "$P $N tokens, pre vs new" | tee -a $R/identity/summary.txt
  done
done

echo "== 4. chunk invariance: greedy ids of the 4k prompt (48 tokens) for several --prefill-chunk"
for B in new pre; do
  BIN=$NEW; [ $B = pre ] && BIN=$PRE
  for C in 128 256 512 1024 2048; do
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids \
      --prefill-chunk $C --prompt "$(cat $PB/prompt-4k.txt)" > $R/chunk/$B-4k-c$C.out 2> $R/chunk/$B-4k-c$C.err
  done
  for C in 128 512 1024 2048; do
    python3 -I $R/first_divergence.py $R/chunk/$B-4k-c256.err $R/chunk/$B-4k-c$C.err "$B 4k: chunk 256 vs $C" | tee -a $R/chunk/summary.txt
  done
done
for C in 128 256 512 1024 2048; do
  python3 -I $R/first_divergence.py $R/chunk/pre-4k-c$C.err $R/chunk/new-4k-c$C.err "4k chunk $C: pre vs new" | tee -a $R/chunk/summary.txt
done

echo "== 5. accuracy against the fp32 goldens, gate class bf16 (new, then pre)"
for B in new pre; do
  BIN=$NEW; [ $B = pre ] && BIN=$PRE
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/accuracy/$B-short.out 2> $R/accuracy/$B-short.err; echo "accuracy $B short exit $?"
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $R/accuracy/$B-long.out 2> $R/accuracy/$B-long.err; echo "accuracy $B long exit $?"
done
for C in 128 512 2048; do
  $NEW accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --mode prefill --max-seq 4096 --prefill-chunk $C \
    > $R/chunk/acc-new-short-c$C.out 2> $R/chunk/acc-new-short-c$C.err; echo "accuracy prefill new chunk $C exit $?"
done
date >> $R/start.txt
echo done
