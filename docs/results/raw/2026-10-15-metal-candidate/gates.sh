#!/bin/zsh
# metal-candidate gates, ONE fleetq slot. base = spike/metal 6271eec build, cand = metal-candidate (spike/metal + P3 + F1).
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
W=/opt/code/intarsiaml/.claude/worktrees/agent-a0a61a602516c3238
R=$W/docs/results/raw/2026-10-15-metal-candidate/gates
RAW=$W/docs/results/raw
PB=$RAW/2026-10-14-metal-phase-b
FD=$RAW/2026-10-14-metal-prefill-p1/full
BASE=$S/intarsia-base
CAND=$S/intarsia-cand
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/decode $R/identity $R/chunk $R/accuracy $R/plan $R/qualify
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 150 --label "metal-candidate gates (tests, accuracy, ids, qualify)" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
git rev-parse HEAD > $R/versions.txt; ls -l $BASE $CAND >> $R/versions.txt; $CAND --version >> $R/versions.txt 2>&1
pgrep -fl zoom.us > $R/zoom.txt 2>&1
echo "== 1. metal unit tests + kernel admission"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/cargo-test-metal.txt 2>&1; echo "cargo test metal exit $?"
grep -E "^test result" $R/cargo-test-metal.txt
$CAND kernels test --backend metal --device 0 --model $M -v > $R/kernels-test.txt 2>&1; echo "kernels test exit $?"
echo "ADMITTED count: $(grep -c ADMITTED $R/kernels-test.txt)"; grep -E "FAIL|REJECT" $R/kernels-test.txt | head; grep -E "kernels," $R/kernels-test.txt
echo "== 2. plan"
$CAND plan show --model $M --backend metal --device 0 --json > $R/plan/cand-default.json 2>$R/plan/cand-default.err
$CAND plan show --model $M --backend metal --device 0 --json --prefill-chunk 256 > $R/plan/cand-c256.json 2>$R/plan/cand-c256.err
echo "plan cand default: $(grep -o '"prefill_chunk[a-z_]*": *[^,]*' $R/plan/cand-default.json | tr '\n' ' ')"
echo "== 3. decode dispatches per step (364 expected), base vs cand ABBA"
for P in short 4k; do
  for B in base cand cand base; do
    BIN=$CAND; [ $B = base ] && BIN=$BASE
    N=$(ls $R/decode | grep -c "^$B-$P-.*jsonl"); F=$R/decode/$B-$P-$((N+1))
    INTARSIA_METAL_PROFILE=cb:$F.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $PB/prompt-$P.txt)" > $F.out 2> $F.err
    echo "$B $P: period/busy/idle/disp $(python3 -I $PB/b2b/period.py $F.jsonl 364 128 2>&1 | tail -1) $(grep -o '([0-9.]* tok/s)' $F.err | tail -1)"
  done
done
echo "== 4. accuracy vs fp32 goldens (bf16 gate class): base (256), cand default (2048), cand --prefill-chunk 256"
for B in base cand cand-c256; do
  BIN=$CAND; X=(); [ $B = base ] && BIN=$BASE; [ $B = cand-c256 ] && X=(--prefill-chunk 256)
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 $X > $R/accuracy/$B-short.out 2> $R/accuracy/$B-short.err; echo "accuracy $B short exit $?"
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 $X > $R/accuracy/$B-long.out 2> $R/accuracy/$B-long.err; echo "accuracy $B long exit $?"
  for L in short long; do echo "$B $L: $(grep -E 'floor \[bf16\]|PASS|FAIL' $R/accuracy/$B-$L.out | grep -oE 'top-1 [0-9.]+%|full-vocab KL [0-9.e-]+|(PASS|FAIL)' | tr '\n' ' ')"; grep -h "prefill chunk" $R/accuracy/$B-$L.err | sort -u | head -2; done
done
echo "== 4b. baseline self-check against P3's recorded Metal baselines/envelopes (read-only, no re-record), cand default and c256"
SB=$W/golden/baselines
for C in default 256; do
  X=(); [ $C != default ] && X=(--prefill-chunk $C)
  $CAND accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --baseline $SB/qwen3-8b-fp32.metal.baseline.json --noise-envelope $SB/qwen3-8b-fp32.metal.envelope.json --allow-same-code --pre-change e8be186 $X > $R/accuracy/selfcheck-short-c$C.out 2> $R/accuracy/selfcheck-short-c$C.err; echo "selfcheck short $C exit $?"
  grep -E "verdict" $R/accuracy/selfcheck-short-c$C.out | cut -c1-300
  $CAND accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --mode both --decode-positions 64 --max-seq 8192 --baseline $SB/qwen3-8b-long-fp32.metal.baseline.json --noise-envelope $SB/qwen3-8b-long-fp32.metal.envelope.json --allow-same-code --pre-change e8be186 $X > $R/accuracy/selfcheck-long-c$C.out 2> $R/accuracy/selfcheck-long-c$C.err; echo "selfcheck long $C exit $?"
  grep -E "verdict" $R/accuracy/selfcheck-long-c$C.out | cut -c1-300
done
echo "== 5. greedy ids, base vs cand"
for N in 48 256; do for P in short 4k; do
  for B in base cand; do
    BIN=$CAND; [ $B = base ] && BIN=$BASE
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids --prompt "$(cat $PB/prompt-$P.txt)" > $R/identity/$B-$P-$N.out 2> $R/identity/$B-$P-$N.err
  done
  python3 -I $FD/first_divergence.py $R/identity/base-$P-$N.err $R/identity/cand-$P-$N.err "$P $N tokens base vs cand" | tee -a $R/identity/summary.txt
done; done
echo "== 6. chunk invariance (cand): 4k 48 ids default vs several --prefill-chunk; short-golden prefill KL per chunk"
$CAND generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids --prompt "$(cat $PB/prompt-4k.txt)" > $R/chunk/4k-cdefault.out 2> $R/chunk/4k-cdefault.err
for C in 64 128 256 512 1024 2048 4050; do
  $CAND generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids --prefill-chunk $C --prompt "$(cat $PB/prompt-4k.txt)" > $R/chunk/4k-c$C.out 2> $R/chunk/4k-c$C.err
  python3 -I $FD/first_divergence.py $R/chunk/4k-cdefault.err $R/chunk/4k-c$C.err "cand 4k: default vs chunk $C" | tee -a $R/chunk/summary.txt
done
for C in 16 64 128 256 512 2048; do
  $CAND accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --mode prefill --max-seq 4096 --prefill-chunk $C > $R/chunk/acc-short-c$C.out 2> $R/chunk/acc-short-c$C.err
  echo "accuracy prefill cand short chunk $C: $(grep -oE 'full-vocab KL [0-9.e-]+' $R/chunk/acc-short-c$C.out | head -1)"
done
echo "== 7. qualify, default recipe (chunk 2048) and recipe forced to metal chunk 256"
sed 's/^prefill_chunk = 2048/prefill_chunk = 256/' $W/recipes/qwen3.toml > $R/qwen3-c256.toml
diff $W/recipes/qwen3.toml $R/qwen3-c256.toml > $R/recipe-c256.diff
rm -rf $R/qualify/default $R/qualify/c256
$CAND qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $G --out $R/qualify/default > $R/qualify-default-console.log 2>&1; echo "qualify default exit $?"
grep -E "Verdict|PASS|SKIP|FAIL" $R/qualify/default/full-report.md | cut -c1-260 | head -40
$CAND qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $G --recipe $R/qwen3-c256.toml --out $R/qualify/c256 > $R/qualify-c256-console.log 2>&1; echo "qualify c256 exit $?"
grep -E "Verdict|PASS|SKIP|FAIL" $R/qualify/c256/full-report.md | cut -c1-260 | head -40
echo done
