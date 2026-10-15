#!/bin/zsh
# Gates for the NAX_MIN_M 16 -> 2 change, run inside an already-acquired fleetq slot (FLEETQ_JOB set by the caller).
SC=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
FIX=$SC/intarsia-fix; CAND=$SC/final/intarsia-cand
W=/opt/code/intarsiaml-p3-noise
R=$W/docs/results/raw/2026-10-15-metal-p3-noise/gates
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
FD=$W/docs/results/raw/2026-10-14-metal-prefill-p1/full
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/accuracy $R/identity
cd $W
echo "== 1. cargo test"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/cargo-test-metal.txt 2>&1; echo "cargo test metal exit $?"
grep -E "^test result|FAILED|panicked" $R/cargo-test-metal.txt | sort | uniq -c
echo "== 2. kernels test"
$FIX kernels test --backend metal --device 0 --model $M -v > $R/kernels-test.txt 2>&1; echo "kernels test exit $?"
echo "ADMITTED lines: $(grep -c ADMITTED $R/kernels-test.txt)"; grep -E "FAIL|REJECT" $R/kernels-test.txt | head; grep -E "kernels," $R/kernels-test.txt
echo "== 3. accuracy, bf16 gate class, FIX then CAND"
for B in fix cand; do
  eval BIN=\$${(U)B}
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/accuracy/$B-short.out 2> $R/accuracy/$B-short.err; echo "accuracy $B short exit $?"
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $R/accuracy/$B-long.out 2> $R/accuracy/$B-long.err; echo "accuracy $B long exit $?"
  for F in short long; do
    echo "$B $F: $(grep -E 'floor \[bf16\]' $R/accuracy/$B-$F.out | grep -oE '(prefill|decode[^ ]*) floor|top-1 [0-9.]+%|full-vocab KL [0-9.e-]+|=> (PASS|FAIL)' | tr '\n' ' ')"
  done
done
echo "== 4. greedy ids FIX vs CAND"
for P in short 4k; do for N in 48 256; do
  for B in cand fix; do eval BIN=\$${(U)B}
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids \
      --prompt "$(cat $PB/prompt-$P.txt)" > $R/identity/$B-$P-$N.out 2> $R/identity/$B-$P-$N.err
  done
  python3 -I $FD/first_divergence.py $R/identity/cand-$P-$N.err $R/identity/fix-$P-$N.err "$P $N tokens cand vs fix" | tee -a $R/identity/summary.txt
done; done
for K in 4086 4107 4110 4120; do for N in 48 256; do
  for B in cand fix; do eval BIN=\$${(U)B}
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids \
      --prompt "$(cat $SC/prompts/w$K.txt)" > $R/identity/$B-w$K-$N.out 2> $R/identity/$B-w$K-$N.err
  done
  python3 -I $FD/first_divergence.py $R/identity/cand-w$K-$N.err $R/identity/fix-w$K-$N.err "w$K $N tokens cand vs fix" | tee -a $R/identity/summary.txt
done; done
echo GATES_DONE
