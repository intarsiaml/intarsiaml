#!/usr/bin/env bash
# Lane lt, spark side: freeze `continue` sequences for CPU oracles (fx5c recipe): `accuracy --mode reuse --reuse-diag --reuse-dump`
# on the default CUDA build, one prompt at a time (the dump exits after the prompt's continue rows). Inside a fleetq slot (GPU).
# Usage: dump.sh <outname> <tree> <prompt:chunk> ...   writes out/<outname>/<prompt>-c<chunk>/dump/seq.json
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; OUT=$P/out/$1; T=$2; shift 2
export PATH=/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
B=$P/$T/target/acc/release/intarsia
for pc in "$@"; do
  IFS=: read -r pr chunk <<< "$pc"; D=$OUT/$pr-c$chunk; mkdir -p $D
  echo "== dump $pr chunk=$chunk $(date +%T)"
  timeout 1500 $B accuracy --model $M35 --backend cuda --golden $P/pf-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode reuse --max-seq 16384 --mem-budget 40 \
    --allow-dirty --no-reuse-oracle --reuse-diag --reuse-prompt $pr --prefill-chunk $chunk --reuse-dump $D/dump < /dev/null > $D/run.log 2>&1
  echo "exit $?"; grep -a "dump:" $D/run.log | cut -c1-200
done
echo "== done $(date +%T)"
