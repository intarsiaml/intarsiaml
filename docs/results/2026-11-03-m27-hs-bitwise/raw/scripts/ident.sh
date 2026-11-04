#!/bin/zsh
# usage: ident.sh OUTDIR BASEBIN NEWBIN ; greedy 256 ids, base vs new with HSB=3 (and 6, 2), spec off, + spec mtp k2
O=$1; B=$2; N=$3; mkdir -p $O
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4
P2="Write a Rust function that parses a CSV line into fields, handling quoted commas, with unit tests."
PR=docs/results/raw/2026-10-15-metal-longctx-profile/prompts
for pn in short 4k 16k; do
  case $pn in short) PA=(--chat --no-think -p "$P2");; 4k) PA=(--prompt "$(cat $PR/prompt-4096.txt)");; 16k) PA=(--prompt "$(cat $PR/prompt-16384.txt)");; esac
  for mode in off k2; do
    case $mode in off) SP=(--spec off);; k2) SP=(--spec mtp --spec-k 2 --mtp-draft-vocab 65536);; esac
    run() { # name bin envassign
      env ${=3} $2 generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 $SP $PA > $O/$pn-$mode-$1.out 2> $O/$pn-$mode-$1.err
    }
    run base $B X=1
    run hsb3 $N INTARSIA_METAL_ATTN_HSB=3
    run hsb6 $N INTARSIA_METAL_ATTN_HSB=6
    run hsb2 $N INTARSIA_METAL_ATTN_HSB=2
    run hsboff $N X=1
    run hs3 $N INTARSIA_METAL_ATTN_HS=3
    for a in hsb3 hsb6 hsb2 hsboff hs3; do cmp -s $O/$pn-$mode-base.out $O/$pn-$mode-$a.out && echo "ident $pn $mode: base == $a" || echo "ident $pn $mode: base != $a"; done
    echo "$pn $mode: $(grep -h 'finish=' $O/$pn-$mode-base.err | grep -o 'decode [0-9]* ms ([0-9.]* tok/s)') / hsb3 $(grep -h 'finish=' $O/$pn-$mode-hsb3.err | grep -o 'decode [0-9]* ms ([0-9.]* tok/s)')"
  done
done
