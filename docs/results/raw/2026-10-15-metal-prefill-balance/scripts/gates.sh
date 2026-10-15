#!/bin/zsh
# Gates for balanced prefill chunking, run inside an already-acquired exclusive fleetq slot (FLEETQ_JOB set by the caller).
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad
NEW=$SC/bin/intarsia-new; BASE=$SC/bin/intarsia-base
W=/opt/code/intarsiaml-prefill-balance
R=$W/docs/results/raw/2026-10-15-metal-prefill-balance/gates
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
FD=$W/docs/results/raw/2026-10-14-metal-prefill-p1/full
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/accuracy $R/identity $R/qualify
cd $W
echo "== 1. cargo test (workspace; then metal release single-threaded)"
cargo test --workspace > $R/cargo-test-workspace.txt 2>&1; echo "cargo test workspace exit $?"
grep -E "^test result|FAILED|panicked" $R/cargo-test-workspace.txt | sort | uniq -c | sort -rn | head -8
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/cargo-test-metal.txt 2>&1; echo "cargo test metal exit $?"
grep -E "^test result|FAILED|panicked" $R/cargo-test-metal.txt | sort | uniq -c
echo "== 2. greedy ids base vs new through generate (the serve-path planner)"
for P in short 4k; do for N in 48 256; do
  for B in base new; do eval BIN=\$${(U)B}
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids \
      --prompt "$(cat $PB/prompt-$P.txt)" > $R/identity/$B-$P-$N.out 2> $R/identity/$B-$P-$N.err
  done
  python3 -I $FD/first_divergence.py $R/identity/base-$P-$N.err $R/identity/new-$P-$N.err "$P $N tokens base vs new" | tee -a $R/identity/summary.txt
done; done
for K in 4086 4107 4110 4120; do for N in 48 256; do
  for B in base new; do eval BIN=\$${(U)B}
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids \
      --prompt "$(cat $SC/prompts/w$K.txt)" > $R/identity/$B-w$K-$N.out 2> $R/identity/$B-w$K-$N.err
  done
  python3 -I $FD/first_divergence.py $R/identity/base-w$K-$N.err $R/identity/new-w$K-$N.err "w$K $N tokens base vs new" | tee -a $R/identity/summary.txt
done; done
echo "== 3. accuracy, bf16 gate class, base then new (Model::forward splitter: not the serve-path planner)"
for B in base new; do
  eval BIN=\$${(U)B}
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/accuracy/$B-short.out 2> $R/accuracy/$B-short.err; echo "accuracy $B short exit $?"
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $R/accuracy/$B-long.out 2> $R/accuracy/$B-long.err; echo "accuracy $B long exit $?"
  for F in short long; do
    echo "$B $F: $(grep -E 'floor \[bf16\]' $R/accuracy/$B-$F.out | grep -oE '(prefill|decode[^ ]*) floor|top-1 [0-9.]+%|full-vocab KL [0-9.e-]+|=> (PASS|FAIL)' | tr '\n' ' ')"
  done
done
echo "== 4. qualify (fresh, new binary; acc-4k holds the engine prefix-reuse gate past 2 chunks of 2048)"
$NEW qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $G --out $R/qualify > $R/qualify-console.log 2>&1; echo "qualify exit $?"
grep -E "^\[qualify.*(PASS|FAIL|SKIP|ERROR)" $R/qualify-console.log | cut -c1-260
echo GATES_DONE
