#!/bin/zsh
# In-engine tile sweep: kernel-mode GPU trace of a single-chunk prompt with INTARSIA_NAX_TILE forcing one tile for every NAX
# GEMM, rounds interleaved (tile order rotated per round). One fleetq slot.  args: rounds size...
W=/opt/code/intarsiaml-gemm-midm
R=$W/docs/results/raw/2026-10-15-metal-gemm-midm/eng3
BIN=${INTARSIA_BIN:-$W/target/release/intarsia}
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
unset INTARSIA_NAX_PICK
TILES=(t128x64w4 t64x64w4 t64x64w2 t32x128w2)
ROUNDS=$1; shift
pgrep -fl zoom.us > $R/zoom.txt 2>&1
# warm the page cache
$BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 --profile-every 0 --prompt "$(cat $R/prompt-1100.txt)" > /dev/null 2>&1
for S in "$@"; do
  for r in $(seq 1 $ROUNDS); do
    k=$(( (r - 1) % 4 ))
    ORDER=(${TILES[@]:$k} ${TILES[@]:0:$k})
    for T in $ORDER; do
      F=$R/k-$S-$T-r$r
      macmon pipe -i 200 > $F.macmon.jsonl 2>/dev/null &
      MP=$!
      INTARSIA_NAX_TILE=$T INTARSIA_METAL_PROFILE=kernel:$F.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 \
        --max-tokens 1 --profile-every 0 --prefill-chunk 4096 --prompt "$(cat $R/prompt-$S.txt)" > $F.out 2> $F.err
      kill $MP 2>/dev/null
    done
  done
  echo "size $S done"
done
