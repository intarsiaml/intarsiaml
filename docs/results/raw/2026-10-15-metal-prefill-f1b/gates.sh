#!/bin/zsh
# F1b correctness gates. pre = spike/metal 6271eec binary, new = branch metal-f1b-nax-attn tip, f1 = F1 + merge (5c30da0).
# Run inside ONE fleetq slot (FLEETQ_JOB set by the caller).
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/f1b
W=/opt/code/intarsiaml/.claude/worktrees/agent-afe24a73270ed0c31
RAW=$W/docs/results/raw/2026-10-15-metal-prefill-f1b
R=$RAW/gates
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
PC=$W/docs/results/raw/2026-10-14-metal-phase-c
FD=$W/docs/results/raw/2026-10-14-metal-prefill-p1/full
NEW=$S/intarsia-final
PRE=$S/intarsia-base
F1=$S/intarsia-f1
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/select $R/decode $R/identity $R/chunk $R/accuracy $R/tests
cd $W
date > $R/start.txt
pgrep -fl zoom.us > $R/zoom.txt 2>&1

echo "== 1. unit tests and admission"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/tests/cargo-test-metal.txt 2>&1; echo "cargo test metal exit $?"
grep -E "^test result|FAILED|panicked" $R/tests/cargo-test-metal.txt | head -40
cargo test --workspace --release --exclude intarsia-backend-metal > $R/tests/cargo-test-workspace.txt 2>&1; echo "cargo test workspace exit $?"
grep -E "^test result" $R/tests/cargo-test-workspace.txt | awk '{p+=$4; f+=$6} END {print "workspace: passed", p, "failed", f}'
cargo check --workspace > $R/tests/cargo-check-workspace.txt 2>&1; echo "cargo check workspace exit $?"
$NEW kernels test --backend metal --device 0 --model $M -v > $R/tests/kernels-test.txt 2>&1; echo "kernels test exit $?"
grep -E "nax|flash|kernels," $R/tests/kernels-test.txt | grep -E "ADMITTED|FAIL|kernels," | cut -c1-220

echo "== 2. kernel selection in the engine (1k prompt, chunk 256)"
for B in pre new new-nometal4; do
  BIN=$NEW; [ $B = pre ] && BIN=$PRE
  [ $B = new-nometal4 ] && export INTARSIA_METAL_NO_METAL4=1
  INTARSIA_METAL_PROFILE=kernel:$R/select/$B-1k.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 \
    --temperature 0 --max-tokens 1 --prompt "$(cat $PC/prompt-1k.txt)" > $R/select/$B-1k.out 2> $R/select/$B-1k.err
  unset INTARSIA_METAL_NO_METAL4
  python3 -I -c "
import json, collections
print('$B prefill 1k kernels:', dict(collections.Counter(json.loads(l)['op'] for l in open('$R/select/$B-1k.jsonl'))))" | tee $R/select/$B-1k-kernels.txt
done
grep -q "attn_prefill_bf16_flash_nax" $R/select/new-1k-kernels.txt || echo "ABORT: NAX attention not selected"
grep -q "attn_prefill_bf16_flash_nax" $R/select/new-nometal4-1k-kernels.txt && echo "ABORT: NAX attention selected without metal4"
grep -q "misaligned operands" $R/select/new-1k.err && echo "ABORT: NAX attention fell back to P2 (misaligned operands)"

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

echo "== 4. accuracy against the fp32 goldens (bf16 gate class): pre, f1, new"
for B in pre f1 new; do
  BIN=$NEW; [ $B = pre ] && BIN=$PRE; [ $B = f1 ] && BIN=$F1
  $RAW/acc.sh gate-$B $BIN
  cp $RAW/acc/gate-$B-short.out $R/accuracy/$B-short.out; cp $RAW/acc/gate-$B-long.out $R/accuracy/$B-long.out
done

echo "== 5. greedy ids, pre vs new"
for N in 48 256; do
  for P in short 4k; do
    for B in pre new; do
      BIN=$NEW; [ $B = pre ] && BIN=$PRE
      $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids \
        --prompt "$(cat $PB/prompt-$P.txt)" > $R/identity/$B-$P-$N.out 2> $R/identity/$B-$P-$N.err
    done
    python3 -I $FD/first_divergence.py $R/identity/pre-$P-$N.err $R/identity/new-$P-$N.err "$P $N tokens, pre vs new" | tee -a $R/identity/summary.txt
  done
done

echo "== 6. chunk invariance (new, then pre): 4k prompt ids (48 tokens) for several --prefill-chunk, and prefill KL of the short golden"
for B in new pre; do
  BIN=$NEW; [ $B = pre ] && BIN=$PRE
  for C in 128 256 1024 2048 4050; do
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids \
      --prefill-chunk $C --prompt "$(cat $PB/prompt-4k.txt)" > $R/chunk/$B-4k-c$C.out 2> $R/chunk/$B-4k-c$C.err
  done
  for C in 128 1024 2048 4050; do
    python3 -I $FD/first_divergence.py $R/chunk/$B-4k-c256.err $R/chunk/$B-4k-c$C.err "$B 4k: chunk 256 vs $C" | tee -a $R/chunk/summary.txt
  done
  for C in 128 256 512 2048; do
    $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --mode prefill --max-seq 4096 --prefill-chunk $C \
      > $R/chunk/acc-$B-short-c$C.out 2> $R/chunk/acc-$B-short-c$C.err
    echo "accuracy prefill $B chunk $C: $(grep -oE 'full-vocab KL [0-9.e-]+' $R/chunk/acc-$B-short-c$C.out | head -1)"
  done
done

echo "== 7. qualify (full, fresh output dir, never --resume)"
rm -rf $R/qualify
$NEW qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $G --out $R/qualify > $R/qualify-console.log 2>&1; echo "qualify exit $?"
grep -E "Verdict|Accuracy|Prefill|Decode|PASS|SKIP|FAIL" $R/qualify/full-report.md | cut -c1-260 | head -40

echo "== 8. the constant-V test must fail on the F1 kernel (unrounded l sum) and pass on the new one"
cp $W/kernels/metal/attention_flash_nax.metal $S/attention_flash_nax.new.metal
git show 5c30da0:kernels/metal/attention_flash_nax.metal > $W/kernels/metal/attention_flash_nax.metal
cargo test --release -p intarsia-backend-metal --test flash_prefill_nax -- --test-threads=1 flash_constant_v > $R/tests/constant-v-on-f1-kernel.txt 2>&1; echo "constant-V test on the F1 kernel: exit $? (non-zero expected)"
grep -E "^test |panicked|got" $R/tests/constant-v-on-f1-kernel.txt | head -5 | cut -c1-200
cp $S/attention_flash_nax.new.metal $W/kernels/metal/attention_flash_nax.metal
git status --short kernels/ | head -3
cargo test --release -p intarsia-backend-metal --test flash_prefill_nax -- --test-threads=1 flash_constant_v > $R/tests/constant-v-on-new-kernel.txt 2>&1; echo "constant-V test on the new kernel: exit $?"
date >> $R/start.txt
echo done
