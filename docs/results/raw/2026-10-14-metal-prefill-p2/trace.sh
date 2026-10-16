#!/bin/zsh
# P2 prefill traces: GPU-timestamp kernel traces (one command buffer per launch) at 4k and 16k, pre vs new, plus plain prefill timing.
# usage: trace.sh <outdir> <pre-binary> <new-binary>   (run inside a fleetq slot)
O=$1; PRE=$2; NEW=$3
R=/opt/code/intarsiaml/.claude/worktrees/agent-a6a0312aabf864646/docs/results/raw
M=/Users/josef/models/Qwen3-8B
mkdir -p $O
for i in 1 2 3; do ioreg -r -d 1 -w0 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tee $O/gpu-idle-before.txt
gen() { # binary prompt-file max-seq extra...
  local B=$1 PF=$2 MS=$3; shift 3
  $B generate --model $M --backend metal --max-seq $MS --temperature 0 --max-tokens 1 --prompt "$(cat $PF)" --profile-every 0 "$@"
}
P4=$R/2026-10-14-metal-phase-b/prompt-4k.txt
P16=$R/2026-10-14-metal-phase-b/b2a/prompt-16k.txt
gen $NEW $P4 8192 > $O/warm.out 2> $O/warm.err   # page-in / pipeline warm-up, discarded
for L in pre new; do
  B=$NEW; [ $L = pre ] && B=$PRE
  gen $B $P4 8192 > $O/plain-4k-$L-r1.out 2> $O/plain-4k-$L-r1.err
  gen $B $P4 8192 > $O/plain-4k-$L-r2.out 2> $O/plain-4k-$L-r2.err
  INTARSIA_METAL_PROFILE=kernel:$O/kernel-4k-$L-c256.jsonl gen $B $P4 8192 > $O/kernel-4k-$L.out 2> $O/kernel-4k-$L.err
  gen $B $P16 17408 > $O/plain-16k-$L.out 2> $O/plain-16k-$L.err
  INTARSIA_METAL_PROFILE=kernel:$O/kernel-16k-$L-c256.jsonl gen $B $P16 17408 > $O/kernel-16k-$L.out 2> $O/kernel-16k-$L.err
done
gzip -f $O/*.jsonl
echo done
