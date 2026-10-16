#!/bin/zsh
# cb-mode (sum of command-buffer GPU time) GPU busy and prefill wall time at chunk 256 vs 2048, interleaved, idle-gated. One slot.
#   cb-chunk.sh <rounds> <sizes...>
W=/opt/code/intarsiaml-gemm-midm
R=$W/docs/results/raw/2026-10-15-metal-gemm-midm/cbchunk
BIN=${INTARSIA_BIN:-$W/target/release/intarsia}
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
source $W/docs/results/raw/2026-10-15-metal-gemm-midm/waitidle.sh
unset INTARSIA_NAX_PICK INTARSIA_NAX_TILE
ROUNDS=$1; shift
for S in "$@"; do
  for r in $(seq 1 $ROUNDS); do
    ORDER="256 2048 2048 256"; [ $((r % 2)) = 0 ] && ORDER="2048 256 256 2048"
    j=0
    for C in ${=ORDER}; do
      j=$((j+1))
      F=$R/cb-$S-c$C-r$r-$j
      wait_idle || touch $F.contaminated
      macmon pipe -i 200 > $F.macmon.jsonl 2>/dev/null &
      MP=$!
      INTARSIA_METAL_PROFILE=cb:$F.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 \
        --profile-every 0 --prefill-chunk $C --prompt "$(cat $R/prompt-$S.txt)" > $F.out 2> $F.err
      kill $MP 2>/dev/null
    done
  done
done
echo done
