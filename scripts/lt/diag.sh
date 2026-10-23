#!/usr/bin/env bash
# Lane lt, spark side: `intarsia accuracy --mode reuse --reuse-diag` on one prompt of the 35B long golden, per arm. Splits a
# failing `continue` row into its causes (decoded vs prefilled reply rows, chunk boundaries). Inside a fleetq slot.
# One process per prompt (the restore diag after a prompt's rows cannot roll the recurrent state back and ends the run).
# Usage: diag.sh <outname> <prompt[,prompt...]> <arm> ...   arm = name:tree:chunk[:ENV=V,ENV=V]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; OUT=$P/out/$1; PR=$2; shift 2
export PATH=/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
G=$P/pf-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
mkdir -p $OUT
for spec in "$@"; do
  IFS=: read -r name tree chunk envs <<< "$spec"; envs=${envs//,/ }
  B=$P/$tree/target/acc/release/intarsia
  for pr in ${PR//,/ }; do
  echo "== diag $name $pr chunk=$chunk env='$envs' $(date +%T)"
  env $envs ${EXTRA_ENV:-} timeout 1500 $B accuracy --model $M35 --backend cuda --golden $G --mode reuse --max-seq 16384 --mem-budget 40 \
    --prefill-chunk $chunk --allow-dirty --reuse-prompt $pr --reuse-diag --reuse-measure ${EXTRA_ARGS:-} < /dev/null > $OUT/diag-$name-$pr.log 2>&1
  echo "exit $?"; grep -aE "vs |REUSE-MEASURE|divergence|continue|restore" $OUT/diag-$name-$pr.log | cut -c1-260
  done
done
echo "== done $(date +%T)"
