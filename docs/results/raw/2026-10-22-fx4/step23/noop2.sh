#!/bin/zsh
# fx4 Metal no-op: origin/main (659f770c) vs lane-fx4/rope-bias on Qwen3-8B. Run inside ONE exclusive fleetq slot on endor.
SP=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4
F=$SP/noop2
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
PB=/opt/code/intarsiaml/docs/results/raw/2026-10-14-metal-phase-b
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $F
cp /opt/code/wt-fx4-base-target/release/intarsia $F/intarsia-base
cp /opt/code/wt-fx4-target/release/intarsia $F/intarsia-new
for B in base new; do
  BIN=$F/intarsia-$B
  $BIN plan show --model $M --backend metal --json > $F/plan-$B.json 2> $F/plan-$B.err; echo "plan $B exit $?"
  $BIN plan show --model $M --backend metal > $F/plan-$B.txt 2>> $F/plan-$B.err; echo "plan(text) $B exit $?"
  $BIN kernels test --backend metal --model $M > $F/kt-$B.log 2>&1; echo "kernels $B exit $? :: $(tail -1 $F/kt-$B.log)"
  for P in short 4k; do
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids \
      --prompt "$(cat $PB/prompt-$P.txt)" > $F/ids-$B-$P.out 2> $F/ids-$B-$P.err; echo "ids $B $P exit $?"
  done
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $F/acc-$B-short.out 2> $F/acc-$B-short.err; echo "acc $B short exit $?"
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $F/acc-$B-long.out 2> $F/acc-$B-long.err; echo "acc $B long exit $?"
done
echo "=== verdicts ==="
cmp $F/plan-base.json $F/plan-new.json && echo "PLAN JSON IDENTICAL"
cmp $F/plan-base.txt $F/plan-new.txt && echo "PLAN TEXT IDENTICAL"
for P in short 4k; do cmp $F/ids-base-$P.out $F/ids-new-$P.out && echo "IDS $P IDENTICAL ($(wc -c < $F/ids-new-$P.out) bytes)"; done
for P in short long; do diff <(grep -v -i -E "elapsed|seconds| ms|tok/s" $F/acc-base-$P.out) <(grep -v -i -E "elapsed|seconds| ms|tok/s" $F/acc-new-$P.out) > /dev/null && echo "ACC $P IDENTICAL"; done
