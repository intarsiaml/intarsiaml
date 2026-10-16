#!/bin/zsh
# Kernel-mode GPU traces (one command buffer per launch) of 4k and 16k prefill at chunk 256 and 2048 for the binaries given
# as label=path arguments, plus the analysis table (kernel TFLOPS). Run inside ONE fleetq slot.
#   trace.sh <outdir-name> label=binary [label=binary ...]
W=/opt/code/intarsiaml/.claude/worktrees/agent-afe24a73270ed0c31
PC=$W/docs/results/raw/2026-10-14-metal-phase-c
P16=$W/docs/results/raw/2026-10-14-metal-phase-b/b2a/prompt-16k.txt
M=/Users/josef/models/Qwen3-8B
OUT=$W/docs/results/raw/2026-10-15-metal-prefill-f1b/$1
shift
mkdir -p $OUT
export INTARSIA_CPU_FREE_MEM_GB=60
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
promptfile() { if [ $1 = 16k ]; then echo $P16; else echo $PC/prompt-$1.txt; fi; }
pgrep -fl zoom.us > $OUT/zoom.txt 2>&1
echo "idle: $(idle3)" | tee $OUT/idle.txt
for spec in "$@"; do
  L=${spec%%=*}
  B=${spec#*=}
  for C in 256 2048; do
    for P in 4k 16k; do
      D=$OUT/$L-c$C
      mkdir -p $D
      INTARSIA_METAL_PROFILE=kernel:$D/kernel-$P-c$C.jsonl $B generate --model $M --backend metal --max-seq 20480 --temperature 0 \
        --max-tokens 1 --profile-every 0 --prefill-chunk $C --prompt "$(cat $(promptfile $P))" > $D/kernel-$P-c$C.out 2> $D/kernel-$P-c$C.err
      python3 -I $W/docs/results/raw/2026-10-14-metal-prefill-p2/analyze_prefill.py $D $P $C > $D/analysis-$P-c$C.txt 2>&1
      echo "$L c$C $P: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $D/kernel-$P-c$C.err | tail -1)"
      cat $D/analysis-$P-c$C.txt | head -30
    done
  done
done
echo "idle end: $(idle3)" | tee -a $OUT/idle.txt
