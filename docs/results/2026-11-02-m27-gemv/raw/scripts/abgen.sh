#!/bin/zsh
# usage: abgen.sh OUTDIR PROMPTFILE TOKENS ROUNDS ARM...   ARM = name:BIN[:ENV=VAL]   (interleaved rounds, MTP off, 4k prompt)
O=$1; F=$2; T=$3; R=$4; shift 4; mkdir -p $O
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4
PR="$(cat $F)"
ARMS=("$@"); N=${#ARMS[@]}
for r in $(seq 1 $R); do for i in $(seq 0 $((N-1))); do arm=${ARMS[$(( (i + r - 1) % N + 1 ))]}
  name=${arm%%:*}; rest=${arm#*:}; bin=${rest%%:*}; envv=""; [[ $rest == *:* ]] && envv=${rest#*:}
  if [[ $name == *k2* ]]; then SP=(--spec mtp --spec-k 2 --mtp-draft-vocab 65536); else SP=(--spec off); fi
  env ${(s:,:)envv} $bin generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens $T --ignore-eos --profile-every 0 $SP --prompt "$PR" > $O/r$r-$name.out 2> $O/r$r-$name.err
  echo "r$r $name: $(grep -h 'finish=' $O/r$r-$name.err | grep -o 'prefill [0-9]* ms ([0-9.]* tok/s) | decode [0-9]* ms ([0-9.]* tok/s)')"
done; done
