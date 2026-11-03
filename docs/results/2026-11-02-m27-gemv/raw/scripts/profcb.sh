#!/bin/zsh
# usage: profcb.sh OUTPREFIX PROMPT BIN...  -> cb-mode profile (normal submission) of 40 decode tokens, per binary
O=$1; F=$2; shift 2
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4
for bin in "$@"; do n=$(basename $bin)
  INTARSIA_METAL_PROFILE=cb:$O-$n.jsonl $bin generate --model $M --backend metal --max-seq 5000 --temperature 0 --max-tokens 40 --ignore-eos --profile-every 0 --spec off --prompt "$(cat $F)" > $O-$n.out 2> $O-$n.err
  echo "$n: $(grep -h 'finish=' $O-$n.err | grep -o 'decode [0-9]* ms ([0-9.]* tok/s)')"
done
