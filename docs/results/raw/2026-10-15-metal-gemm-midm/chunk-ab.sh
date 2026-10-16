#!/bin/zsh
# In-engine kernel-mode traces of a 4k prefill at chunk 256 vs 2048 (same binary, same default pick), rounds interleaved, idle-gated,
# macmon GPU clock per run. One fleetq slot.  args: rounds
W=/opt/code/intarsiaml-gemm-midm
R=$W/docs/results/raw/2026-10-15-metal-gemm-midm/chunk
BIN=${INTARSIA_BIN:-$W/target/release/intarsia}
M=/Users/josef/models/Qwen3-8B
source $W/docs/results/raw/2026-10-15-metal-gemm-midm/waitidle.sh
export INTARSIA_CPU_FREE_MEM_GB=60
unset INTARSIA_NAX_PICK INTARSIA_NAX_TILE
pgrep -fl zoom.us > $R/zoom.txt 2>&1
for r in $(seq 1 $1); do
  ORDER="256 2048"; [ $((r % 2)) = 0 ] && ORDER="2048 256"
  for C in ${=ORDER}; do
    F=$R/k-4062-c$C-r$r
    wait_idle || touch $F.contaminated
    macmon pipe -i 200 > $F.macmon.jsonl 2>/dev/null &
    MP=$!
    INTARSIA_METAL_PROFILE=kernel:$F.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 \
      --profile-every 0 --prefill-chunk $C --prompt "$(cat $R/prompt-4062.txt)" > $F.out 2> $F.err
    kill $MP 2>/dev/null
  done
done
echo done
