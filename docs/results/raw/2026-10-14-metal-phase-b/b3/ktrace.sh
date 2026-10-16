#!/bin/zsh
# usage: ktrace.sh <label=bin> ...  : kernel-mode GPU trace (129-token decode, short prompt), idle-utilization gated
R=/opt/code/intarsiaml/docs/results/raw/2026-10-14-metal-phase-b
O=$R/b3/profile
M=/Users/josef/models/Qwen3-8B
mkdir -p $O
util() { ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; }
for spec in "$@"; do
  label=${spec%%=*}; bin=${spec#*=}
  u=$(util); tries=0; while [ "${u:-99}" -gt 10 ] && [ $tries -lt 30 ]; do sleep 2; u=$(util); tries=$((tries+1)); done
  echo "$label idle_util=$u(waits=$tries)"
  INTARSIA_METAL_PROFILE=kernel:$O/$label-kernel-short.jsonl $bin generate --model $M --backend metal --max-seq 8192 \
    --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $R/prompt-short.txt)" \
    > $O/$label-kernel-short.out 2> $O/$label-kernel-short.err
done
