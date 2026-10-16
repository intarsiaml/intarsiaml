#!/bin/zsh
# In-engine A/B of the pick rules: kernel-mode GPU trace of a single-chunk prompt, arms p45 (before), new (shipped default),
# pad (fewest padded rows), rounds interleaved with the arm order rotated. One fleetq slot.  args: rounds size...
W=/opt/code/intarsiaml-gemm-midm
R=$W/docs/results/raw/2026-10-15-metal-gemm-midm/eng5
BIN=${INTARSIA_BIN:-$W/target/release/intarsia}
M=/Users/josef/models/Qwen3-8B
source $W/docs/results/raw/2026-10-15-metal-gemm-midm/waitidle.sh
export INTARSIA_CPU_FREE_MEM_GB=60
unset INTARSIA_NAX_TILE
ARMS=(p45 new)
ROUNDS=$1; shift
pgrep -fl zoom.us > $R/zoom.txt 2>&1
$BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 --profile-every 0 --prompt "$(cat $R/prompt-1000.txt)" > /dev/null 2>&1
for S in "$@"; do
  for r in $(seq 1 $ROUNDS); do
    k=$(( (r - 1) % 2 ))
    ORDER=(${ARMS[@]:$k} ${ARMS[@]:0:$k})
    for A in $ORDER; do
      F=$R/k-$S-$A-r$r
      if [ $A = new ]; then unset INTARSIA_NAX_PICK; else export INTARSIA_NAX_PICK=$A; fi
      wait_idle || touch $F.contaminated
      macmon pipe -i 200 > $F.macmon.jsonl 2>/dev/null &
      MP=$!
      INTARSIA_METAL_PROFILE=kernel:$F.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 \
        --max-tokens 1 --profile-every 0 --prefill-chunk 4096 --prompt "$(cat $R/prompt-$S.txt)" > $F.out 2> $F.err
      kill $MP 2>/dev/null
    done
  done
  echo "size $S done"
done
