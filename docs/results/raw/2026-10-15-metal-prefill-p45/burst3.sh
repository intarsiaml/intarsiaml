#!/bin/zsh
# burst 3: stream traces (cb mode: GPU idle between command buffers), qk lever isolation at 4k c2048 / c256, qkv proxy bench
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p45
R=$W/docs/results/raw/2026-10-15-metal-prefill-p45
PC=$W/docs/results/raw/2026-10-14-metal-phase-c
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/cb
for C in 256 2048; do
  for B in pre t2; do
    INTARSIA_METAL_PROFILE=cb:$R/cb/cb-$B-4k-c$C.jsonl $S/intarsia-$B generate --model $M --backend metal --max-seq 8192 --temperature 0 \
      --max-tokens 1 --profile-every 0 --prefill-chunk $C --prompt "$(cat $PC/prompt-4k.txt)" > $R/cb/cb-$B-4k-c$C.out 2> $R/cb/cb-$B-4k-c$C.err
    echo "cb $B c$C: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $R/cb/cb-$B-4k-c$C.err | tail -1)"
  done
done
$R/plain-abba.sh plain-t2-qk-c2048 2048 4k 2 A=$S/intarsia-t2@INTARSIA_METAL_NO_QK_PF=1 B=$S/intarsia-t2
$R/plain-abba.sh plain-t2-qk-c256 256 4k 2 A=$S/intarsia-t2@INTARSIA_METAL_NO_QK_PF=1 B=$S/intarsia-t2
