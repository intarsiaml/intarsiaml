#!/bin/zsh
# P1 + P2 together (branch merged with spike/metal 05f0b8b, which has the P2 flash prefill attention): correctness burst.
# pre = spike/metal 05f0b8b (P2 only, built from `git archive`), new = branch metal-p1-gemm (P1 + P2). Run in a fleetq slot.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
W=/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7
R=$W/docs/results/raw/2026-10-14-metal-prefill-p1/full/p1p2
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
PC=$W/docs/results/raw/2026-10-14-metal-phase-c
FD=$W/docs/results/raw/2026-10-14-metal-prefill-p1/full
NEW=$S/intarsia-new
PRE=$S/intarsia-pre
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/select $R/decode $R/identity $R/chunk $R/accuracy $R/tests
cd $W

echo "== 0. pre revision: $(cat $S/pre-rev.txt)"
echo "== 1. unit tests and admission on the merged tree"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/tests/cargo-test-metal.txt 2>&1; echo "cargo test exit $?"
$NEW kernels test --backend metal --device 0 --model $M -v > $R/tests/kernels-test.txt 2>&1; echo "kernels test exit $?"
grep -E "nax|flash|kernels," $R/tests/kernels-test.txt | grep -E "ADMITTED|FAIL|kernels,"

echo "== 2. kernel selection in the engine (abort if NAX or flash is not used in prefill)"
for B in pre new; do
  BIN=$NEW; [ $B = pre ] && BIN=$PRE
  INTARSIA_METAL_PROFILE=kernel:$R/select/$B-1k.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 \
    --temperature 0 --max-tokens 1 --prompt "$(cat $PC/prompt-1k.txt)" > $R/select/$B-1k.out 2> $R/select/$B-1k.err
  python3 -I -c "
import json, collections
print('$B prefill 1k kernels:', dict(collections.Counter(json.loads(l)['op'] for l in open('$R/select/$B-1k.jsonl'))))" | tee $R/select/$B-1k-kernels.txt
done
grep -q "matmul_bf16_nax" $R/select/new-1k-kernels.txt && grep -q "matmul_pair_bf16_nax" $R/select/new-1k-kernels.txt && grep -q "attn_prefill_bf16_flash" $R/select/new-1k-kernels.txt || { echo "ABORT: NAX or flash not selected"; exit 1; }

echo "== 3. decode dispatches per step (364 expected)"
for P in short 4k; do
  for B in pre new; do
    BIN=$NEW; [ $B = pre ] && BIN=$PRE
    F=$R/decode/$B-$P
    INTARSIA_METAL_PROFILE=cb:$F.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 \
      --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $PB/prompt-$P.txt)" > $F.out 2> $F.err
    echo "$B $P: $(python3 -I $PB/b2b/period.py $F.jsonl 364 128 2>&1 | tail -1) $(grep -o '([0-9.]* tok/s)' $F.err | tail -1)"
  done
done

echo "== 4. accuracy against the fp32 goldens (bf16 gate class), new and pre"
for B in new pre; do
  BIN=$NEW; [ $B = pre ] && BIN=$PRE
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/accuracy/$B-short.out 2> $R/accuracy/$B-short.err; echo "accuracy $B short exit $?"
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $R/accuracy/$B-long.out 2> $R/accuracy/$B-long.err; echo "accuracy $B long exit $?"
  echo "$B short: $(grep -oE 'top-1 [0-9.]+%|full-vocab KL [0-9.e-]+|: (PASS|FAIL)' $R/accuracy/$B-short.out | tr '\n' ' ')"
  echo "$B long : $(grep -oE 'top-1 [0-9.]+%|full-vocab KL [0-9.e-]+|: (PASS|FAIL)' $R/accuracy/$B-long.out | tr '\n' ' ')"
done

echo "== 5. greedy ids, pre (P2 only) vs new (P1 + P2)"
for N in 48 256; do
  for P in short 4k; do
    for B in pre new; do
      BIN=$NEW; [ $B = pre ] && BIN=$PRE
      $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids \
        --prompt "$(cat $PB/prompt-$P.txt)" > $R/identity/$B-$P-$N.out 2> $R/identity/$B-$P-$N.err
    done
    python3 -I $FD/first_divergence.py $R/identity/pre-$P-$N.err $R/identity/new-$P-$N.err "$P $N tokens" | tee -a $R/identity/summary.txt
  done
done

echo "== 6. chunk invariance (new binary, then pre): 4k prompt ids for several --prefill-chunk, and prefill KL of the short golden"
for B in new pre; do
  BIN=$NEW; [ $B = pre ] && BIN=$PRE
  for C in 128 256 512 1024 4050; do
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids \
      --prefill-chunk $C --prompt "$(cat $PB/prompt-4k.txt)" > $R/chunk/$B-4k-c$C.out 2> $R/chunk/$B-4k-c$C.err
  done
  for C in 128 512 1024 4050; do
    python3 -I $FD/first_divergence.py $R/chunk/$B-4k-c256.err $R/chunk/$B-4k-c$C.err "$B 4k: chunk 256 vs $C" | tee -a $R/chunk/summary.txt
  done
  for C in 128 256 512; do
    $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --mode prefill --max-seq 4096 --prefill-chunk $C \
      > $R/chunk/acc-$B-short-c$C.out 2> $R/chunk/acc-$B-short-c$C.err
    echo "accuracy prefill $B chunk $C: $(grep -oE 'full-vocab KL [0-9.e-]+' $R/chunk/acc-$B-short-c$C.out | head -1)"
  done
done

echo "== 7. qualify (full, with goldens)"
$NEW qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $G --out $R/qualify > $R/qualify-console.log 2>&1; echo "qualify exit $?"
grep -E "Verdict|Accuracy|Prefill|Decode" $R/qualify/full-report.md | cut -c1-260
echo done
