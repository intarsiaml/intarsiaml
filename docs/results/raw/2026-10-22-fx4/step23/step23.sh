#!/bin/zsh
# fx4 steps 2+3, one exclusive endor slot: Qwen3-8B no-op (main vs branch), Qwen2.5 accuracy before/after, 4k prefill profiles
SP=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4
PB=/opt/code/intarsiaml/docs/results/raw/2026-10-14-metal-phase-b
export INTARSIA_CPU_FREE_MEM_GB=60
$SP/noop2.sh
mkdir -p $SP/s23
FINAL=$SP/noop2/intarsia-new          # bias single-rounding + flash groups 3,5,6,7
BIAS=$SP/bin/intarsia-bias-only        # bias single-rounding only
ONB=$SP/bin/intarsia-onb               # first branch build: double-rounded bias
cd /opt/code/wt-fx4
G=/Users/josef/models/golden
for V in onb bias final; do
  case $V in onb) B=$ONB;; bias) B=$BIAS;; final) B=$FINAL;; esac
  for N in qwen2.5-7b-instruct:Qwen2.5-7B-Instruct llama-3.1-8b-instruct:Llama-3.1-8B-Instruct; do
    g=${N%%:*}; d=${N##*:}
    [ $V != final ] && [ $g = llama-3.1-8b-instruct ] && continue   # Llama has no bias: only the final binary matters
    $B accuracy --model /Users/josef/models/$d --backend metal --golden $G/$g-fp32.safetensors --gate-class bf16 > $SP/s23/acc-$g-$V.out 2> $SP/s23/acc-$g-$V.err; echo "acc $g $V exit $?"
  done
done
for V in onb final; do
  case $V in onb) B=$ONB;; final) B=$FINAL;; esac
  for N in Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct; do
    $B generate --model /Users/josef/models/$N --backend metal --max-seq 8192 --profile-every 1 --op-table --temperature 0 --max-tokens 1 --prompt "$(cat $PB/prompt-4k.txt)" > $SP/s23/prof-$N-$V.out 2> $SP/s23/prof-$N-$V.err; echo "profile $N $V exit $?"
  done
done
