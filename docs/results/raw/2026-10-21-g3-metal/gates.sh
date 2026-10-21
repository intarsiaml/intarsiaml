#!/bin/zsh
# G3 GPU gates, one exclusive slot: base = origin/main 4be7b7d4 build, dev = perf/metal-g3 build (defaults).
#  1 Metal kernels test (dev)   2 cargo test -p intarsia-backend-metal --release (real model on)
#  3 greedy ids 256 tokens, prompt-short and prompt-4k, base vs dev byte-identical (stdout incl. ids)
#  4 goldens: accuracy qwen3-8b-fp32 and qwen3-8b-long-fp32, base vs dev logs byte-identical
#  5 qualify (dev)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; mkdir -p $O
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad
W=/opt/code/intarsiaml/.claude/worktrees/agent-a34836167f72ea687
R=$W/docs/results/raw
P=$R/2026-10-15-metal-longctx-profile/prompts
PS=$R/2026-10-15-metal-decode-d1/prompt-short.txt
M=/Users/josef/models/Qwen3-8B; G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60 DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
B=$S/intarsia-base; D=$S/intarsia-dev
echo "base $(md5 -q $B) dev $(md5 -q $D) $(date -Iseconds)" | tee $O/info.txt
echo "== kernels test"
$D kernels test --backend metal --device 0 --model $M -v > $O/kernels-test.txt 2>&1; echo "kernels test exit $?"; tail -2 $O/kernels-test.txt
echo "== cargo test metal"
( cd $W && INTARSIA_TEST_REAL_MODEL=1 cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $O/cargo-test-metal.txt 2>&1 ); echo "cargo test metal exit $?"
grep -h "test result" $O/cargo-test-metal.txt | awk '{p+=$4; f+=$6} END {print "total passed", p, "failed", f}'
echo "== greedy ids 256"
for f in short:$PS 4k:$P/prompt-4096.txt; do
  n=${f%%:*}; pf=${f#*:}
  for a in base dev; do
    BIN=$B; [ $a = dev ] && BIN=$D
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids \
      --prompt "$(cat $pf)" > $O/gen256-$n-$a.out 2> $O/gen256-$n-$a.err; echo "gen $n $a exit $?"
  done
  cmp $O/gen256-$n-base.out $O/gen256-$n-dev.out && echo "greedy 256 $n: byte-identical"
done
echo "== goldens"
for a in base dev; do
  BIN=$B; [ $a = dev ] && BIN=$D
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 > $O/acc-short-$a.log 2>&1; echo "acc short $a exit $?"
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --mode both --decode-positions 16 --max-seq 8192 > $O/acc-long-$a.log 2>&1; echo "acc long $a exit $?"
done
for k in short long; do cmp -s $O/acc-$k-base.log $O/acc-$k-dev.log && echo "acc $k: logs byte-identical" || { echo "acc $k: DIFFER"; diff $O/acc-$k-base.log $O/acc-$k-dev.log | head; }; done
grep -hiE "kl|top-1|pass|fail" $O/acc-short-dev.log $O/acc-long-dev.log | tail -12
echo "== qualify (dev)"
( cd $W && $D qualify --model $M --backend metal --golden-dir $G --out $O/qualify > $O/qualify.txt 2>&1 ); echo "qualify exit $?"; tail -25 $O/qualify.txt
echo GATES_DONE $(date -Iseconds)
