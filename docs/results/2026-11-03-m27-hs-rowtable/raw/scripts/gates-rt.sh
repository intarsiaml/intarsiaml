#!/bin/zsh
# Gates for the row-table lane: NEWBIN = .scratch/intarsia.rt, BASEBIN = .scratch/intarsia.def (env unset). usage: gates-rt.sh OUTROOT NEWBIN BASEBIN (from the worktree root, inside a fleetq slot)
O=$1; F=$2; B=$3; mkdir -p $O/identity $O/kernels $O/plan-show $O/accuracy $O/mtp
unset INTARSIA_METAL_ATTN_HS INTARSIA_METAL_ATTN_HSB
W=/Users/josef/models; M=$W/Qwen3.8-27B-Quark-AWQ-MXFP4
P2="Write a Rust function that parses a CSV line into fields, handling quoted commas, with unit tests."
PR=docs/results/raw/2026-10-15-metal-longctx-profile/prompts
echo "== 1/5 greedy ids + text: base vs rt (env unset) vs opt-out HSB=0"
for pn in short 4k 16k; do
  case $pn in short) PA=(--chat --no-think -p "$P2");; 4k) PA=(--prompt "$(cat $PR/prompt-4096.txt)");; 16k) PA=(--prompt "$(cat $PR/prompt-16384.txt)");; esac
  for mode in off k2; do
    case $mode in off) SP=(--spec off);; k2) SP=(--spec mtp --spec-k 2 --mtp-draft-vocab 65536);; esac
    run() { env ${=3} $2 generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 --show-ids $SP $PA > $O/identity/$pn-$mode-$1.out 2> $O/identity/$pn-$mode-$1.err; grep -E '^ids' $O/identity/$pn-$mode-$1.err > $O/identity/$pn-$mode-$1.ids; }
    run base $B X=1
    run rt $F X=1
    run optout $F INTARSIA_METAL_ATTN_HSB=0
    for a in rt optout; do
      cmp -s $O/identity/$pn-$mode-base.out $O/identity/$pn-$mode-$a.out && t=same || t=DIFF
      cmp -s $O/identity/$pn-$mode-base.ids $O/identity/$pn-$mode-$a.ids && i=same || i=DIFF
      echo "ident $pn $mode: base vs $a: text $t, ids $i ($(tr ',' '\n' < $O/identity/$pn-$mode-base.ids | wc -l | tr -d ' ') ids)"
    done
  done
done
echo "== 2/5 kernels test, default binary vs base, no env"
$B kernels test --backend metal > $O/kernels/base-full.log 2>&1; echo "base kernels test exit $?"
$F kernels test --backend metal > $O/kernels/rt-full.log 2>&1; echo "def kernels test exit $?"
for op in attn_decode attn_prefill; do
  $B kernels test --backend metal -m $M --op $op -v > $O/kernels/m27-$op-base.log 2>&1
  $F kernels test --backend metal -m $M --op $op -v > $O/kernels/m27-$op-rt.log 2>&1
  cmp -s $O/kernels/m27-$op-base.log $O/kernels/m27-$op-rt.log && echo "27B-config $op: logs identical" || echo "27B-config $op: logs DIFFER"
done
echo "== 3/5 plan show"
for mdl in Qwen3.6-35B-A3B-NVFP4 Qwen3-8B Qwen3.8-27B-Quark-AWQ-MXFP4; do for fmt in "" "--json"; do
  $B plan show --model $W/$mdl --backend metal $fmt > $O/plan-show/$mdl.main$fmt.txt 2> $O/plan-show/$mdl.main$fmt.err
  $F plan show --model $W/$mdl --backend metal $fmt > $O/plan-show/$mdl.rt$fmt.txt 2> $O/plan-show/$mdl.rt$fmt.err
  cmp -s $O/plan-show/$mdl.main$fmt.txt $O/plan-show/$mdl.rt$fmt.txt && echo "plan show $mdl $fmt: identical" || echo "plan show $mdl $fmt: DIFFERS"
done; done
echo "== 4/5 accuracy"
G=docs/results/2026-11-02-m27-metal/raw/golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
BL=golden/baselines/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.metal.baseline.json
$F accuracy --backend metal --model $M --max-seq 4096 --mxfp4-act none --golden $G --mode both --decode-positions 16 --baseline $BL --allow-unstamped > $O/accuracy/rt.out 2> $O/accuracy/rt.err; echo "accuracy exit $?"
grep -E "verdict|golden gate" $O/accuracy/rt.out
cmp -s $O/accuracy/rt.out docs/results/2026-11-03-m27-hs-bitwise/raw/gates-def/accuracy/def.out && echo "accuracy: byte-identical to the previous lane's default-binary run (m27-hs-bitwise gates-def/accuracy/def.out)" || echo "accuracy: DIFFERS from m27-hs-bitwise gates-def/accuracy/def.out"
cmp -s $O/accuracy/rt.out docs/results/2026-11-02-m27-gemv/raw/accuracy/final.out && echo "accuracy: byte-identical to m27-gemv final.out" || echo "accuracy: DIFFERS from m27-gemv final.out"
echo "== 5/5 MTP k2 == off (default binary)"
P1="Write a short story about a lighthouse keeper who discovers a message in a bottle. Include dialogue."
for pi in 1 2; do eval P=\$P$pi
  for mode in off k2; do case $mode in off) A=(--spec off);; k2) A=(--spec mtp --spec-k 2 --mtp-draft-vocab 65536);; esac
    $F generate --model $M --backend metal --max-seq 4096 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 $A --chat --no-think -p "$P" > $O/mtp/p$pi-$mode.out 2> $O/mtp/p$pi-$mode.err
  done
  cmp $O/mtp/p$pi-off.out $O/mtp/p$pi-k2.out && echo "mtp p$pi: k2 == off"
done
