#!/bin/zsh
# Gates for engine/localfail on endor. usage (inside a fleetq GPU slot, from the worktree root): gates-lf.sh OUTROOT NEWBIN BASEBIN
set -u
O=${1:A}; F=${2:A}; B=${3:A}; mkdir -p $O/{plan-show,kernels,identity,serve,accuracy}
unset INTARSIA_METAL_ATTN_HS INTARSIA_METAL_ATTN_HSB
W=/Users/josef/models; M=$W/Qwen3.8-27B-Quark-AWQ-MXFP4; M8=$W/Qwen3-8B
P2="Write a Rust function that parses a CSV line into fields, handling quoted commas, with unit tests."
PF=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt
echo "== 1/5 plan show (text and --json), base vs new"
for mdl in Qwen3.6-35B-A3B-NVFP4 Qwen3-8B Qwen3.8-27B-Quark-AWQ-MXFP4; do for fmt in "" "--json"; do
  $B plan show --model $W/$mdl --backend metal $fmt > $O/plan-show/$mdl.base$fmt.txt 2> $O/plan-show/$mdl.base$fmt.err
  $F plan show --model $W/$mdl --backend metal $fmt > $O/plan-show/$mdl.new$fmt.txt 2> $O/plan-show/$mdl.new$fmt.err
  cmp -s $O/plan-show/$mdl.base$fmt.txt $O/plan-show/$mdl.new$fmt.txt && echo "plan show $mdl $fmt: identical ($(wc -c < $O/plan-show/$mdl.new$fmt.txt | tr -d ' ') bytes)" || echo "plan show $mdl $fmt: DIFFERS"
done; done
echo "== 2/5 kernels test --backend metal"
$B kernels test --backend metal > $O/kernels/base.log 2>&1; echo "base kernels test exit $?"
$F kernels test --backend metal > $O/kernels/new.log 2>&1; echo "new kernels test exit $?"
echo "FAIL lines: base $(grep -c FAIL $O/kernels/base.log) new $(grep -c FAIL $O/kernels/new.log)"; tail -3 $O/kernels/new.log
diff $O/kernels/base.log $O/kernels/new.log > $O/kernels/diff.txt && echo "kernels logs identical" || { echo "kernels logs differ:"; cat $O/kernels/diff.txt; }
echo "== 3/5 single-stream generate ids: 27B and 8B, short and 4K, --spec off, base vs new"
for mn in 27b 8b; do case $mn in 27b) MD=$M;; 8b) MD=$M8;; esac
  for pn in short 4k; do
    case $pn in short) PA=(--chat --no-think -p "$P2");; 4k) PA=(--prompt "$(cat $PF)");; esac
    for t in base new; do case $t in base) X=$B;; new) X=$F;; esac
      $X generate --model $MD --backend metal --max-seq 6000 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 --show-ids --spec off $PA > $O/identity/$mn-$pn-$t.out 2> $O/identity/$mn-$pn-$t.err
      grep -E '^ids' $O/identity/$mn-$pn-$t.err > $O/identity/$mn-$pn-$t.ids
    done
    cmp -s $O/identity/$mn-$pn-base.out $O/identity/$mn-$pn-new.out && tx=same || tx=DIFF
    cmp -s $O/identity/$mn-$pn-base.ids $O/identity/$mn-$pn-new.ids && ix=same || ix=DIFF
    echo "ident $mn $pn: text $tx, ids $ix ($(tr ',' '\n' < $O/identity/$mn-$pn-base.ids | wc -l | tr -d ' ') base ids, $(tr ',' '\n' < $O/identity/$mn-$pn-new.ids | wc -l | tr -d ' ') new ids)"
  done; done
echo "== 5/5 accuracy 8B goldens (new vs base byte-compare)"
G=/opt/code/intarsiaml/golden
for g in qwen3-8b-fp32 qwen3-8b-long-fp32; do
  case $g in *long*) DP=64;; *) DP=16;; esac
  for t in base new; do case $t in base) X=$B;; new) X=$F;; esac
    $X accuracy --backend metal --model $M8 --max-seq 4096 --golden $G/$g.safetensors --mode both --decode-positions $DP --baseline golden/baselines/$g.metal.baseline.json --allow-unstamped > $O/accuracy/$g-$t.out 2> $O/accuracy/$g-$t.err; echo "accuracy $g $t exit $?"
  done
  grep -E "verdict|golden gate" $O/accuracy/$g-new.out | head -4
  cmp -s $O/accuracy/$g-base.out $O/accuracy/$g-new.out && echo "accuracy $g: base and new byte-identical" || echo "accuracy $g: DIFFERS"
done
echo "== done"
