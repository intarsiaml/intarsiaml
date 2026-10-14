#!/bin/zsh
# usage: ab.sh <short|4k> <pairs> <binA-label=path> <binB-label=path> [outdir]
# Interleaved A/B of the decode stream (A B, B A, A B, ...): cb-mode GPU trace of a 129-token forced decode (128 steps),
# step period / GPU busy / idle in ms per step from period.py, plus the engine's own tok/s. Run inside a fleetq slot.
# Before each run the GPU's idle "Device Utilization %" is sampled (ioreg); above 10 it waits and resamples.
CTX=$1; PAIRS=$2; A=$3; B=$4
R=/opt/code/intarsiaml/docs/results/raw/2026-10-14-metal-phase-b
O=${5:-$R/b4/ab}
M=/Users/josef/models/Qwen3-8B
D=364   # dispatches per decode step of the fold binary; the pre binary (label "pre") has 544
mkdir -p $O
util() { ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; }
run() {
  local label=${1%%=*} bin=${1#*=} rep=$2
  local u tries=0
  u=$(util); while [ "${u:-99}" -gt 10 ] && [ $tries -lt 20 ]; do sleep 2; u=$(util); tries=$((tries+1)); done
  local d=$D; [ $label = pre ] && d=544
  local f=$O/$label-$CTX-$rep
  INTARSIA_METAL_PROFILE=cb:$f.jsonl $bin generate --model $M --backend metal --max-seq 8192 --profile-every 0 \
    --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $R/prompt-$CTX.txt)" > $f.out 2> $f.err
  echo "$label $CTX pair$rep idle_util=$u(waits=$tries) $(python3 $R/b2b/period.py $f.jsonl $d 128) $(grep -o '([0-9.]* tok/s)' $f.err | tail -1)"
}
for i in $(seq 1 $PAIRS); do
  if [ $((i % 2)) = 1 ]; then run $A $i; run $B $i; else run $B $i; run $A $i; fi
done
