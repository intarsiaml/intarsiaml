#!/bin/sh
W=/opt/code/intarsiaml/.worktrees/nab2
P="$(cat $W/raw/nab2/longprompt.txt)"
for C in default 512 100; do
  if [ $C = default ]; then X=""; else X="--prefill-chunk $C"; fi
  $W/target/release/intarsia generate --model /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b --backend metal --max-seq 4096 $X --prompt "$P" --max-tokens 64 --temperature 0 --show-ids > $W/raw/nab2/chunk-$C.txt 2>&1
done
