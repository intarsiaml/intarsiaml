#!/bin/zsh
# Job E: cold `intarsia generate` (fresh process, 4062-token prompt, 1 token), cb-mode trace: base vs v3 vs v3 with each change off.
# cold.sh <outdir> <rounds>
O=$1; N=$2; mkdir -p $O
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/cbgap
R=/opt/code/intarsiaml-cbgap/docs/results/raw/2026-10-16-metal-prefill-cbgap
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
source $R/waitidle.sh
run() { # name binary chunk size [env...]
  local name=$1 bin=$2 C=$3 S=$4; shift 4
  local F=$O/$name-$S-c$C
  wait_idle || touch $F.contaminated
  macmon pipe -i 200 > $F.macmon.jsonl 2>/dev/null & local MP=$!
  env "$@" INTARSIA_METAL_PROFILE=cb:$F.trace.jsonl $SC/$bin generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 \
    --profile-every 0 --prefill-chunk $C --prompt "$(cat $R/prompt-$S.txt)" > $F.out 2> $F.err
  kill $MP 2>/dev/null
}
for r in $(seq 1 $N); do for C in 256 2048; do
  run base-r$r intarsia-base $C 4062 A=1
  run v3-r$r intarsia-v3 $C 4062 A=1
  run v3noprewire-r$r intarsia-v3 $C 4062 INTARSIA_METAL_PREWIRE=0
  run v3nostage-r$r intarsia-v3 $C 4062 INTARSIA_METAL_AHEAD=0
done; done
echo done
