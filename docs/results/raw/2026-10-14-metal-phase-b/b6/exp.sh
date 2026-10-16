#!/bin/zsh
# usage: exp.sh <ctx short|4k> <pairs> <outdir> <label=bin[:ENV=1,ENV2=1:dispatches]> ...
# Interleaved A/B (rotating order each pair) of the decode stream: cb-mode GPU trace of a 129-token forced decode (128 steps),
# step period / GPU busy / idle in ms per step (b2b/period.py) plus the engine's tok/s. Run inside a fleetq slot.
# GPU idle "Device Utilization %" is sampled before each run; above 10 it waits and resamples (max 20 tries).
CTX=$1; PAIRS=$2; O=$3; shift 3
R=/opt/code/intarsiaml/.claude/worktrees/charming-rhodes-7d1cb4/docs/results/raw/2026-10-14-metal-phase-b
M=/Users/josef/models/Qwen3-8B
mkdir -p $O
util() { ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; }
run() {
  local spec=$1 rep=$2
  local label=${spec%%=*}; local rest=${spec#*=}
  local bin=${rest%%:*}; local extra=${rest#*:}; [ "$extra" = "$rest" ] && extra=""
  local envs=${extra%%:*}; local d=${extra#*:}; [ -z "$extra" ] && d=364 && envs="B5_DUMMY=1"
  local u tries=0
  u=$(util); while [ "${u:-99}" -gt 10 ] && [ $tries -lt 20 ]; do sleep 2; u=$(util); tries=$((tries+1)); done
  local f=$O/$label-$CTX-$rep
  local -a ev
  ev=("${(@s:,:)envs}")
  env "${ev[@]}" INTARSIA_METAL_PROFILE=cb:$f.jsonl $bin generate --model $M --backend metal --max-seq 8192 --profile-every 0 \
    --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $R/prompt-$CTX.txt)" > $f.out 2> $f.err
  echo "$label $CTX pair$rep idle_util=$u(waits=$tries) $(python3 $R/b2b/period.py $f.jsonl $d 128) | win(median min mean) $(python3 $R/b6/period2.py $f.jsonl $d 128 16) $(grep -o '([0-9.]* tok/s)' $f.err | tail -1)"
}
specs=("$@")
n=${#specs[@]}
for i in $(seq 1 $PAIRS); do
  for k in $(seq 0 $((n-1))); do
    idx=$(( (k + i - 1) % n + 1 ))
    run "${specs[$idx]}" $i
  done
done
