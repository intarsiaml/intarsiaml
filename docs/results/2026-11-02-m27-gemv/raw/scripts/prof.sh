#!/bin/zsh
# usage: prof.sh BIN OUT PROMPTFILE  -> kernel-mode profile of 24 decode tokens after the prompt
BIN=$1; O=$2; F=$3
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4
INTARSIA_METAL_PROFILE=kernel:$O.jsonl $BIN generate --model $M --backend metal --max-seq 5000 --temperature 0 --max-tokens 24 --ignore-eos --profile-every 0 --spec off --prompt "$(cat $F)" > $O.out 2> $O.err
