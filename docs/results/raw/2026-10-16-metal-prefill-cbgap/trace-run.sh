#!/bin/zsh
# One cbgap-bench process with a cb-mode trace and a macmon log, idle-gated.
# trace-run.sh <cbgap-bench> <out prefix> <chunk> <lens> <reps> <warmup> [VAR=value ...]
B=$1; P=$2; C=$3; L=$4; N=$5; W=$6; shift 6
R=/opt/code/intarsiaml-cbgap/docs/results/raw/2026-10-16-metal-prefill-cbgap
export INTARSIA_CPU_FREE_MEM_GB=60
source $R/waitidle.sh
wait_idle || touch $P.contaminated
macmon pipe -i 200 > $P.macmon.jsonl 2>/dev/null & MP=$!
env "$@" INTARSIA_METAL_PROFILE=cb:$P.trace.jsonl $B --model /Users/josef/models/Qwen3-8B --backend metal --max-seq 8192 --profile-every 0 \
  --prefill-chunk $C --docs /opt/code/intarsiaml-cbgap/tools/prompts/corpus_docs.txt --lens $L --reps $N --warmup $W --out $P.req.jsonl > $P.log 2>&1
kill $MP 2>/dev/null
