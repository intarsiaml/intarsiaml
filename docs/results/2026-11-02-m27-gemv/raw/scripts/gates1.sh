#!/bin/zsh
# gates 1: plan show vs main, 27B accuracy vs the lane's golden baseline, MTP k2 vs off, greedy identity. usage: gates1.sh OUTROOT FINALBIN BASEBIN MAINBIN
O=$1; F=$2; B=$3; MAIN=$4; mkdir -p $O/plan-show $O/accuracy $O/mtp $O/identity
W=/Users/josef/models; M=$W/Qwen3.8-27B-Quark-AWQ-MXFP4
for mdl in Qwen3.6-35B-A3B-NVFP4 Qwen3-8B; do for fmt in "" "--json"; do
  $MAIN plan show --model $W/$mdl --backend metal $fmt > $O/plan-show/$mdl.main$fmt.txt 2> $O/plan-show/$mdl.main$fmt.err
  $F    plan show --model $W/$mdl --backend metal $fmt > $O/plan-show/$mdl.final$fmt.txt 2> $O/plan-show/$mdl.final$fmt.err
  cmp $O/plan-show/$mdl.main$fmt.txt $O/plan-show/$mdl.final$fmt.txt && echo "plan show $mdl $fmt: identical"
done; done
G=docs/results/2026-11-02-m27-metal/raw/golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
BL=golden/baselines/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.metal.baseline.json
$F accuracy --backend metal --model $M --max-seq 4096 --mxfp4-act none --golden $G --mode both --decode-positions 16 --baseline $BL --allow-unstamped > $O/accuracy/final.out 2> $O/accuracy/final.err; echo "accuracy exit $?"
P1="Write a short story about a lighthouse keeper who discovers a message in a bottle. Include dialogue."
P2="Write a Rust function that parses a CSV line into fields, handling quoted commas, with unit tests."
for pi in 1 2; do eval P=\$P$pi
  for mode in off k2; do case $mode in off) A=(--spec off);; k2) A=(--spec mtp --spec-k 2 --mtp-draft-vocab 65536);; esac
    $F generate --model $M --backend metal --max-seq 4096 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 $A --chat --no-think -p "$P" > $O/mtp/p$pi-$mode.out 2> $O/mtp/p$pi-$mode.err
  done
  cmp $O/mtp/p$pi-off.out $O/mtp/p$pi-k2.out && echo "mtp p$pi: k2 == off"
done
# greedy identity at 256 tokens, short prompt and 4k prompt: base, final with the attention split off (gemv + fusion only), final with fusion off too, final default
PR=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt
for pn in short 4k; do
  if [ $pn = short ]; then PA=(--chat --no-think -p "$P2"); else PA=(--prompt "$(cat $PR)"); fi
  $B generate --model $M --backend metal --max-seq 5000 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 --spec off $PA > $O/identity/$pn-base.out 2>/dev/null
  INTARSIA_METAL_ATTN_HS=1 $F generate --model $M --backend metal --max-seq 5000 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 --spec off $PA > $O/identity/$pn-final-hs1.out 2>/dev/null
  INTARSIA_METAL_ATTN_HS=1 INTARSIA_METAL_MXFP4_XFUSE=0 $F generate --model $M --backend metal --max-seq 5000 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 --spec off $PA > $O/identity/$pn-final-hs1-nofuse.out 2>/dev/null
  $F generate --model $M --backend metal --max-seq 5000 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 --spec off $PA > $O/identity/$pn-final.out 2>/dev/null
  for a in final-hs1 final-hs1-nofuse final; do cmp -s $O/identity/$pn-base.out $O/identity/$pn-$a.out && echo "identity $pn: base == $a" || echo "identity $pn: base != $a"; done
done
