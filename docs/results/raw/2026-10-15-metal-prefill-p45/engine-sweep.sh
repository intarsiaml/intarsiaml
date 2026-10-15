#!/bin/zsh
# In-engine tile sweep: kernel-mode GPU trace of prompt P at chunk C with INTARSIA_NAX_TILE forcing one tile for every NAX
# GEMM (binary built with all 21 sweep variants). One fleetq slot. args: tile names.
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p45
PC=$W/docs/results/raw/2026-10-14-metal-phase-c
M=/Users/josef/models/Qwen3-8B
OUT=$W/docs/results/raw/2026-10-15-metal-prefill-p45/engine-sweep
mkdir -p $OUT
export INTARSIA_CPU_FREE_MEM_GB=60
pgrep -fl zoom.us > $OUT/zoom.txt 2>&1
date > $OUT/start.txt
# warm the page cache
$S/intarsia-sweep generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 --profile-every 0 --prompt "$(cat $PC/prompt-1k.txt)" > /dev/null 2>&1
for T in "$@"; do
  for cfg in 4k:256 4k:1024 4k:2048 1k:2048 1k:256; do
    P=${cfg%%:*}
    C=${cfg#*:}
    D=$OUT/$T
    mkdir -p $D
    INTARSIA_NAX_TILE=$T INTARSIA_METAL_PROFILE=kernel:$D/kernel-$P-c$C.jsonl $S/intarsia-sweep generate --model $M --backend metal --max-seq 8192 --temperature 0 \
      --max-tokens 1 --profile-every 0 --prefill-chunk $C --prompt "$(cat $PC/prompt-$P.txt)" > $D/kernel-$P-c$C.out 2> $D/kernel-$P-c$C.err
    python3 -I $W/docs/results/raw/2026-10-14-metal-prefill-p2/analyze_prefill.py $D $P $C > $D/analysis-$P-c$C.txt 2>&1
    echo "$T $P c$C: $(grep -A1 '^| matmul' $D/analysis-$P-c$C.txt | head -1)"
  done
done
date >> $OUT/start.txt
