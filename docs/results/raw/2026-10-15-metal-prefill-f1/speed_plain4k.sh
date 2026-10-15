#!/bin/zsh
# F1: plain prefill on prompt-4k, more repetitions than speed_f1.sh (its 4k pairs scattered +-25%): 8 runs per binary and chunk,
# alternating order (pre new new pre new pre pre new). Run inside ONE fleetq slot.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
W=/opt/code/intarsiaml/.claude/worktrees/agent-ae91acc25118dab7a
R=$W/docs/results/raw/2026-10-15-metal-prefill-f1/speed/plain4k
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
NEW=$S/intarsia-new
PRE=$S/intarsia-pre
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
binof() { [ $1 = pre ] && echo $PRE || echo $NEW; }
pgrep -fl zoom.us | tee $R/zoom.txt
$PRE generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 --profile-every 0 --prompt "$(cat $PB/prompt-4k.txt)" > $R/warm.out 2> $R/warm.err
for C in 256 1024; do
  i=0
  for B in pre new new pre new pre pre new; do
    i=$((i+1))
    echo "idle $B c$C #$i: $(idle3)" >> $R/idle.txt
    $(binof $B) generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 --profile-every 0 --prefill-chunk $C \
      --prompt "$(cat $PB/prompt-4k.txt)" > $R/$B-c$C-$i.out 2> $R/$B-c$C-$i.err
    echo "$B 4k chunk $C #$i: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $R/$B-c$C-$i.err | tail -1)"
  done
done
echo done
