#!/bin/bash
# Per-knob KL on short (16 and 64 decode positions) + self-golden stats vs the knobs-off self-golden of window 2. In a fleetq slot.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
L=/opt/code/intarsiaml/.worktrees/mpre/.lane; C=$L/intarsia-c1; OUT=$L/sc6; W2=$L/w2; mkdir -p $OUT
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; G=/opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
for a in off d256 nv both; do
  case $a in off) E="INTARSIA_METAL_FLASH_NAX_D256=0 INTARSIA_METAL_NVFP4_NAX=0";; d256) E="INTARSIA_METAL_NVFP4_NAX=0";; nv) E="INTARSIA_METAL_FLASH_NAX_D256=0";; both) E="A=1";; esac
  env $E $C accuracy --model $M --backend metal --golden $G --mode both --decode-positions 64 --max-seq 4096 --allow-unstamped > $OUT/short64-$a.log 2>&1
  echo "$a short64 exit $?"
done
for n in short 4k; do
  $C accuracy --model $M --backend metal --mode both --allow-unstamped --allow-same-code --max-seq 8192 --self-golden $W2/$n.off.sg.safetensors > $OUT/self-$n.log 2>&1
  echo "self $n exit $?: $(grep -aiE 'self' $OUT/self-$n.log | tail -3 | cut -c1-300)"
done
