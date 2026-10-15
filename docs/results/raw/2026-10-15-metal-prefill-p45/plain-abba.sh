#!/bin/zsh
# Plain prefill (generate --max-tokens 1) A/B/B/A pairs, one fleetq slot. usage:
#   plain-abba.sh <outdir-name> <chunks, comma> <prompts, comma: 1k,4k,16k> <rounds> A=<binary>[@ENV=VAL] B=<binary>[@ENV=VAL]
# Order per round: A B B A; the first run (page-in) is discarded.
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
PC=$W/docs/results/raw/2026-10-14-metal-phase-c
P16=$W/docs/results/raw/2026-10-14-metal-phase-b/b2a/prompt-16k.txt
M=/Users/josef/models/Qwen3-8B
OUT=$W/docs/results/raw/2026-10-15-metal-prefill-p45/$1
CHUNKS=${2//,/ }
PROMPTS=${3//,/ }
ROUNDS=$4
SPEC_A=$5
SPEC_B=$6
mkdir -p $OUT
export INTARSIA_CPU_FREE_MEM_GB=60
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
pgrep -fl zoom.us > $OUT/zoom.txt 2>&1
echo "idle start: $(idle3)" | tee $OUT/idle.txt
specbin() { echo ${1#*=} | sed 's/@.*//'; }
specenv() { case $1 in *@*) echo ${1#*@};; *) echo "NOENV=1";; esac; }
promptfile() { if [ $1 = 16k ]; then echo $P16; else echo $PC/prompt-$1.txt; fi; }
run() { # label spec chunk prompt idx
  local bin=$(specbin $2) envv=$(specenv $2) pf=$(promptfile $4)
  local f=$OUT/$1-$4-c$3-$5
  env $envv $bin generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 1 --profile-every 0 \
    --prefill-chunk $3 --prompt "$(cat $pf)" > $f.out 2> $f.err
  echo "$1 $4 c$3 #$5: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $f.err | tail -1)"
}
run warm $SPEC_A 2048 4k 0
for C in ${=CHUNKS}; do
  for P in ${=PROMPTS}; do
    echo "idle $P c$C: $(idle3)" >> $OUT/idle.txt
    for r in $(seq 1 $ROUNDS); do
      run A $SPEC_A $C $P ${r}a
      run B $SPEC_B $C $P ${r}b
      run B $SPEC_B $C $P ${r}c
      run A $SPEC_A $C $P ${r}d
    done
  done
done
echo "idle end: $(idle3)" | tee -a $OUT/idle.txt
