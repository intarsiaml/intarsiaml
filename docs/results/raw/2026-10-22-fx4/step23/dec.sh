#!/bin/zsh
SP=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4
PB=/opt/code/intarsiaml/docs/results/raw/2026-10-14-metal-phase-b
for rep in 1 2; do
for V in onb final; do
  case $V in onb) B=$SP/bin/intarsia-onb;; final) B=$SP/noop2/intarsia-new;; esac
  $B generate --model /Users/josef/models/Qwen2.5-7B-Instruct --backend metal --max-seq 8192 --temperature 0 --max-tokens 192 --ignore-eos --prompt "$(cat $PB/prompt-short.txt)" > /dev/null 2> $SP/s23/dec-$V-$rep.err; echo "dec $V $rep: $(grep -E 'finish=' $SP/s23/dec-$V-$rep.err | cut -c1-200)"
  $B generate --model /Users/josef/models/Qwen2.5-7B-Instruct --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 --prompt "$(cat $PB/prompt-4k.txt)" > /dev/null 2> $SP/s23/pf4k-$V-$rep.err; echo "pf4k $V $rep: $(grep -E 'finish=' $SP/s23/pf4k-$V-$rep.err | cut -c1-200)"
done
done
