#!/bin/zsh
# fwdrows at chunk 256 and 2048 (and any extra chunks). Runs inside an exclusive endor fleetq slot.
# run.sh <fwdrows binary> <outdir> <lens> [chunks...]
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
H=$1; O=$2; LENS=$3; shift 3; CHUNKS=${@:-256 2048}
W=/opt/code/intarsiaml-fwdhash; M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $O; cd $W
for C in ${=CHUNKS}; do
  $H --model $M --backend metal --max-seq 8192 --profile-every 0 --prefill-chunk $C --docs tools/prompts/corpus_docs.txt --code tools/prompts/corpus_code.txt \
     --lens $LENS --tail 24 --out $O/c$C.jsonl --dump-dir $O/dump-c$C > $O/c$C.log 2>&1
  echo "fwdrows chunk=$C exit $?"
done
