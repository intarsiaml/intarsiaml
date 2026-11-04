#!/bin/zsh
# usage: ident-final.sh OUTDIR BASEBIN NEWBIN ; the final binary (.scratch/intarsia.hsb2) against base: HSB=3, short / 4k / 16k, MTP off and k2, 256 tokens greedy
O=$1; B=$2; N=$3; mkdir -p $O
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4
P2="Write a Rust function that parses a CSV line into fields, handling quoted commas, with unit tests."
PR=docs/results/raw/2026-10-15-metal-longctx-profile/prompts
for pn in short 4k 16k; do
  case $pn in short) PA=(--chat --no-think -p "$P2");; 4k) PA=(--prompt "$(cat $PR/prompt-4096.txt)");; 16k) PA=(--prompt "$(cat $PR/prompt-16384.txt)");; esac
  for mode in off k2; do
    case $mode in off) SP=(--spec off);; k2) SP=(--spec mtp --spec-k 2 --mtp-draft-vocab 65536);; esac
    env X=1 $B generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 $SP $PA > $O/$pn-$mode-base.out 2> $O/$pn-$mode-base.err
    env INTARSIA_METAL_ATTN_HSB=3 $N generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 $SP $PA > $O/$pn-$mode-hsb3.out 2> $O/$pn-$mode-hsb3.err
    cmp -s $O/$pn-$mode-base.out $O/$pn-$mode-hsb3.out && echo "ident-final $pn $mode: base == hsb3 ($(wc -c < $O/$pn-$mode-base.out) bytes)" || echo "ident-final $pn $mode: base != hsb3"
  done
done
