#!/bin/zsh
# usage: acc-hs.sh OUTROOT FINALBIN HS...  -- the 27B golden accuracy gate against the lane baseline with INTARSIA_METAL_ATTN_HS=<hs>
O=$1; F=$2; shift 2; mkdir -p $O/accuracy
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4
G=docs/results/2026-11-02-m27-metal/raw/golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
BL=golden/baselines/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.metal.baseline.json
for hs in "$@"; do
  INTARSIA_METAL_ATTN_HS=$hs $F accuracy --backend metal --model $M --max-seq 4096 --mxfp4-act none --golden $G --mode both --decode-positions 16 --baseline $BL --allow-unstamped > $O/accuracy/hs$hs.out 2> $O/accuracy/hs$hs.err; echo "hs=$hs accuracy exit $?"
done
