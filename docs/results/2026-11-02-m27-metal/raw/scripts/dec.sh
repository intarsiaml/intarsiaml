#!/bin/zsh
# decode A/B: usage dec.sh OUTDIR BIN_A BIN_B ; 3 interleaved rounds, plain decode (spec off) and MTP k2, 128 tokens
O=$1; A=$2; B=$3; M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; mkdir -p $O
P="Write a Rust function that parses a CSV line into fields, handling quoted commas, with unit tests."
for r in 1 2 3; do for arm in a b; do
  if [ $arm = a ]; then BIN=$A; else BIN=$B; fi
  for mode in off k2; do
    case $mode in off) S=(--spec off);; k2) S=(--spec mtp --spec-k 2 --mtp-draft-vocab 65536);; esac
    $BIN generate --model $M --backend metal --max-seq 4096 --temperature 0 --max-tokens 128 --ignore-eos --profile-every 0 $S --chat --no-think -p "$P" > $O/r$r-$arm-$mode.out 2> $O/r$r-$arm-$mode.err
    echo "r$r $arm $mode: $(grep -h 'finish=' $O/r$r-$arm-$mode.err | grep -o 'decode [0-9]* ms ([0-9.]* tok/s)')"
  done
done; done
