#!/bin/zsh
# Metal gates for the bitwise head split. usage: gates.sh OUTROOT NEWBIN MAINBIN   (run from the worktree root, inside a fleetq slot)
# New binary with INTARSIA_METAL_ATTN_HSB=3 against the main binary (same source, split off) and against its own default.
O=$1; F=$2; MAIN=$3; mkdir -p $O/plan-show $O/accuracy $O/mtp
W=/Users/josef/models; M=$W/Qwen3.8-27B-Quark-AWQ-MXFP4
for mdl in Qwen3.6-35B-A3B-NVFP4 Qwen3-8B; do for fmt in "" "--json"; do
  $MAIN plan show --model $W/$mdl --backend metal $fmt > $O/plan-show/$mdl.main$fmt.txt 2> $O/plan-show/$mdl.main$fmt.err
  INTARSIA_METAL_ATTN_HSB=3 $F plan show --model $W/$mdl --backend metal $fmt > $O/plan-show/$mdl.new$fmt.txt 2> $O/plan-show/$mdl.new$fmt.err
  cmp $O/plan-show/$mdl.main$fmt.txt $O/plan-show/$mdl.new$fmt.txt && echo "plan show $mdl $fmt: identical"
done; done
G=docs/results/2026-11-02-m27-metal/raw/golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
BL=golden/baselines/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.metal.baseline.json
for arm in hsb3 off; do
  if [ $arm = hsb3 ]; then EV=(INTARSIA_METAL_ATTN_HSB=3); else EV=(X=1); fi
  env $EV $F accuracy --backend metal --model $M --max-seq 4096 --mxfp4-act none --golden $G --mode both --decode-positions 16 --baseline $BL --allow-unstamped > $O/accuracy/$arm.out 2> $O/accuracy/$arm.err; echo "accuracy $arm exit $?"
done
P1="Write a short story about a lighthouse keeper who discovers a message in a bottle. Include dialogue."
P2="Write a Rust function that parses a CSV line into fields, handling quoted commas, with unit tests."
for pi in 1 2; do eval P=\$P$pi
  for mode in off k2; do case $mode in off) A=(--spec off);; k2) A=(--spec mtp --spec-k 2 --mtp-draft-vocab 65536);; esac
    INTARSIA_METAL_ATTN_HSB=3 $F generate --model $M --backend metal --max-seq 4096 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 $A --chat --no-think -p "$P" > $O/mtp/p$pi-$mode.out 2> $O/mtp/p$pi-$mode.err
  done
  cmp $O/mtp/p$pi-off.out $O/mtp/p$pi-k2.out && echo "mtp p$pi (HSB=3): k2 == off"
done
