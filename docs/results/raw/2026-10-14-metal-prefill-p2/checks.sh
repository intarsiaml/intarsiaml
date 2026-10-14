#!/bin/zsh
# P2 correctness checks: kernels test, accuracy against the fp32 goldens, greedy-id identity vs the pre binary.
# usage: checks.sh <outdir> <pre-binary> <new-binary> [label:envvalue ...]   (run inside a fleetq slot)
# Each extra "label:variant" runs the new binary again with INTARSIA_METAL_FLASH=<variant> (a flash variant forced).
O=$1; PRE=$2; NEW=$3; shift 3
R=/opt/code/intarsiaml/.claude/worktrees/agent-a6a0312aabf864646/docs/results/raw
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
mkdir -p $O/identity $O/accuracy
export INTARSIA_CPU_FREE_MEM_GB=60
for i in 1 2 3; do ioreg -r -d 1 -w0 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tee $O/gpu-idle-before.txt
$NEW kernels test --backend metal --device 0 --model $M > $O/kernels-test.out 2> $O/kernels-test.err; echo "kernels test exit $?"
runs=("pre:$PRE:" "new:$NEW:" )
for x in "$@"; do runs+=("${x%%:*}:$NEW:${x#*:}"); done
for run in $runs; do
  label=${run%%:*}; rest=${run#*:}; BIN=${rest%%:*}; VAR=${rest#*:}
  if [ -n "$VAR" ]; then export INTARSIA_METAL_FLASH=$VAR; else unset INTARSIA_METAL_FLASH; fi
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $O/accuracy/$label-short.out 2> $O/accuracy/$label-short.err; echo "accuracy $label short exit $?"
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $O/accuracy/$label-long.out 2> $O/accuracy/$label-long.err; echo "accuracy $label long exit $?"
  for P in short 4k; do
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids --prompt "$(cat $R/2026-10-14-metal-phase-b/prompt-$P.txt)" > $O/identity/$label-$P.out 2> $O/identity/$label-$P.err; echo "generate $label $P exit $?"
  done
done
unset INTARSIA_METAL_FLASH
for P in short 4k; do
  for run in $runs; do
    label=${run%%:*}; [ $label = pre ] && continue
    cmp $O/identity/pre-$P.out $O/identity/$label-$P.out && echo "identity $P $label: stdout byte-identical"
    diff <(grep '^ids:' $O/identity/pre-$P.err) <(grep '^ids:' $O/identity/$label-$P.err) > /dev/null && echo "identity $P $label: token ids identical"
  done
done
echo done
