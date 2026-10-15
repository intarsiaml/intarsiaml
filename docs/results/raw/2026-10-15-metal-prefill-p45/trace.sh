#!/bin/zsh
# Kernel-mode GPU traces (one command buffer per launch) of 4k prefill at chunk 256 and 2048 for the binaries given as
# label=path arguments, plus the analysis table. Run inside ONE fleetq slot.
#   trace.sh <outdir-name> label=binary [label=binary ...]
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p45
PC=$W/docs/results/raw/2026-10-14-metal-phase-c
M=/Users/josef/models/Qwen3-8B
OUT=$W/docs/results/raw/2026-10-15-metal-prefill-p45/$1
shift
mkdir -p $OUT
export INTARSIA_CPU_FREE_MEM_GB=60
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
pgrep -fl zoom.us > $OUT/zoom.txt 2>&1
echo "idle: $(idle3)" | tee $OUT/idle.txt
for spec in "$@"; do
  L=${spec%%=*}
  B=${spec#*=}
  for C in 256 2048; do
    for P in 1k 4k; do
      D=$OUT/$L-c$C
      mkdir -p $D
      INTARSIA_METAL_PROFILE=kernel:$D/kernel-$P-c$C.jsonl $B generate --model $M --backend metal --max-seq 8192 --temperature 0 \
        --max-tokens 1 --profile-every 0 --prefill-chunk $C --prompt "$(cat $PC/prompt-$P.txt)" > $D/kernel-$P-c$C.out 2> $D/kernel-$P-c$C.err
      python3 -I $W/docs/results/raw/2026-10-14-metal-prefill-p2/analyze_prefill.py $D $P $C > $D/analysis-$P-c$C.txt 2>&1
      echo "$L c$C $P: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $D/kernel-$P-c$C.err | tail -1) $(grep -A4 '^| class' $D/analysis-$P-c$C.txt | tr '\n' ' ')"
    done
  done
done
echo "idle end: $(idle3)" | tee -a $OUT/idle.txt
