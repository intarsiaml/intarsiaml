#!/bin/zsh
# GPU busy (cb mode: sum of command-buffer GPU times) of a prefill, base (INTARSIA_NAX_PICK=p45) vs new, ABBA rounds, one slot.
#   cb-abba.sh <chunk> <rounds> <sizes...>
W=/opt/code/intarsiaml-gemm-midm
R=$W/docs/results/raw/2026-10-15-metal-gemm-midm/cb
BIN=${INTARSIA_BIN:-$W/target/release/intarsia}
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
source $W/docs/results/raw/2026-10-15-metal-gemm-midm/waitidle.sh
C=$1; ROUNDS=$2; shift 2
pgrep -fl zoom.us > $R/zoom.txt 2>&1
for S in "$@"; do
  for r in $(seq 1 $ROUNDS); do
    ORDER="base new new base"; [ $((r % 2)) = 0 ] && ORDER="new base base new"
    j=0
    for B in ${=ORDER}; do
      j=$((j+1))
      wait_idle || touch $R/cb-$B-$S-c$C-r$r-$j.contaminated
      case $B in base) export INTARSIA_NAX_PICK=p45;; *) unset INTARSIA_NAX_PICK;; esac
      F=$R/cb-$B-$S-c$C-r$r-$j
      macmon pipe -i 200 > $F.macmon.jsonl 2>/dev/null &
      MP=$!
      INTARSIA_METAL_PROFILE=cb:$F.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 \
        --profile-every 0 --prefill-chunk $C --prompt "$(cat $R/prompt-$S.txt)" > $F.out 2> $F.err
      kill $MP 2>/dev/null
      echo "$B $S c$C r$r: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $F.err | tail -1) busy $(python3 -I -c "
import json
rs=[json.loads(l) for l in open('$F.jsonl')]
print(round(sum(r['e']-r['s'] for r in rs)*1000,1),'ms in',len(rs),'cbs; span',round((rs[-1]['e']-rs[0]['s'])*1000,1))")"
    done
  done
done
